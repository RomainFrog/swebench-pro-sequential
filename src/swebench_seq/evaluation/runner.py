"""Single-task evaluation API used by the orchestration layer after rollout.

Public:
  - score_one(repo_dir, instance_id, patch, tag_lookup) -> bool | None
      End-to-end: ensure image, eval, merge result into
      `repo_dir/score/eval_results.json` atomically. Returns True/False
      for pass/fail or None for eval-infra failure.
  - load_tag_lookup(dataset_path) -> dict[str, str]
      Cached SWE-bench Pro `instance_id -> dockerhub_tag` map for the selected
      dataset. Expensive to build, so call once per dataset source.

The upstream evaluator (`SWE-bench_Pro-os/swe_bench_pro_eval.py`) is
invoked via subprocess against Podman's Docker-compatible socket; that's an
environmental requirement, not something we can provide in pure Python.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import docker  # type: ignore[import-untyped]

from ..constants import IMAGE_PREFIX_DEFAULT as _IMAGE_PREFIX
from ..runner import ContainerError, PodmanCli
from .assets import resolve_evaluator_dir
from .dataset import load_swebench_pro_rows, materialize_dataset_jsonl

log = logging.getLogger(__name__)

_MERGE_LOCKS_GUARD = threading.Lock()
_MERGE_LOCKS: dict[Path, threading.Lock] = {}


def _merge_lock(score_dir: Path) -> threading.Lock:
    key = score_dir.resolve()
    with _MERGE_LOCKS_GUARD:
        lock = _MERGE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _MERGE_LOCKS[key] = lock
        return lock


# The upstream evaluator (`SWE-bench_Pro-os/swe_bench_pro_eval.py`) hard-codes
# `--dockerhub_username jefzda` and looks up images by `jefzda/sweap-images:<tag>`
# (Docker SDK normalizes that to `docker.io/jefzda/sweap-images:<tag>`). When
# our `image_prefix` points anywhere else, the upstream cannot find the image
# locally even though we just pulled it under a different name. We therefore
# add the upstream's expected local tag for the duration of grading.
_UPSTREAM_IMAGE_PREFIX = "docker.io/jefzda/sweap-images"
_EVAL_METADATA = "eval_metadata.json"


def _resolve_evaluator_paths(evaluator_dir: Path | None) -> tuple[Path, Path, Path]:
    repo = resolve_evaluator_dir(evaluator_dir)
    script = repo / "swe_bench_pro_eval.py"
    scripts_dir = repo / "run_scripts"
    return repo, script, scripts_dir


def _podman_docker_host(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    configured = os.environ.get("DOCKER_HOST")
    if configured:
        return configured
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return f"unix://{Path(runtime_dir) / 'podman' / 'podman.sock'}"
    return f"unix:///run/user/{os.getuid()}/podman/podman.sock"


def _evaluator_environment(docker_host: str) -> dict[str, str]:
    """Build a minimal environment without forwarding model credentials."""

    allowed = (
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env["DOCKER_HOST"] = docker_host
    # Keep the verified evaluator cache immutable during execution. Importing
    # helper_code otherwise creates __pycache__ entries and invalidates the
    # content hash checked on the next run.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def validate_evaluation_environment(
    evaluator_dir: Path | None,
    *,
    docker_host: str | None = None,
    podman: PodmanCli | None = None,
) -> tuple[Path, str]:
    """Fail fast unless the public evaluator and Podman API are usable."""

    eval_repo, _eval_script, _scripts_dir = _resolve_evaluator_paths(evaluator_dir)
    client = podman or PodmanCli()
    try:
        client.version()
    except (ContainerError, OSError) as exc:
        raise RuntimeError(
            "Podman CLI is unavailable; install Podman and ensure the configured "
            "SWEBENCH_PODMAN_COMMAND works"
        ) from exc

    resolved_host = _podman_docker_host(docker_host)
    api_client = None
    try:
        api_client = docker.APIClient(base_url=resolved_host, timeout=5)
        if not api_client.ping():
            raise RuntimeError("the API ping returned false")
        version_info = api_client.version()
        if not _docker_api_reports_podman(version_info):
            raise RuntimeError(
                "DOCKER_HOST answered but does not identify itself as Podman; "
                "point DOCKER_HOST at the rootless Podman API socket"
            )
    except Exception as exc:
        raise RuntimeError(
            "Podman's Docker-compatible API is unavailable; start the Podman "
            f"service/socket and set DOCKER_HOST to it ({exc})"
        ) from exc
    finally:
        if api_client is not None:
            api_client.close()
    return eval_repo, resolved_host


def _docker_api_reports_podman(version_info: object) -> bool:
    if not isinstance(version_info, Mapping):
        return False
    names: list[str] = []
    platform = version_info.get("Platform")
    if isinstance(platform, Mapping) and isinstance(platform.get("Name"), str):
        names.append(platform["Name"])
    components = version_info.get("Components")
    if isinstance(components, list):
        names.extend(
            component["Name"]
            for component in components
            if isinstance(component, Mapping) and isinstance(component.get("Name"), str)
        )
    return any("podman" in name.lower() for name in names)


def _patch_sha256(patch: str) -> str:
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def _resolved_score(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        resolved = value.get("resolved", value.get("passed"))
        return resolved if isinstance(resolved, bool) else None
    return None


@lru_cache(maxsize=4)
def load_tag_lookup(dataset_path: Path | None = None) -> dict[str, str]:
    """instance_id → dockerhub_tag, from the SWE-bench Pro dataset.

    Cached: the HF `load_dataset` call takes several seconds. Inline
    scoring re-uses the same lookup across every task in the sweep, so
    one cache entry suffices.
    """
    return {
        row["instance_id"]: row["dockerhub_tag"]
        for row in load_swebench_pro_rows(dataset_path)
    }


def _ensure_image(
    tag: str,
    image_prefix: str = _IMAGE_PREFIX,
    *,
    podman: PodmanCli | None = None,
) -> bool:
    """Make `tag` available in the configured Podman store under both names:
    `{image_prefix}:{tag}` (so our orchestrator finds it) and
    `_UPSTREAM_IMAGE_PREFIX:{tag}` (so the upstream eval finds it without
    trying to pull from docker.io).
    """
    client = podman or PodmanCli()
    full = f"{image_prefix}:{tag}"
    upstream_full = f"{_UPSTREAM_IMAGE_PREFIX}:{tag}"

    def _alias_for_upstream() -> None:
        """Ensure upstream_full points at the same image as full. No-op if
        image_prefix already IS the upstream prefix, or if upstream_full
        already exists (e.g. from a previous run or from an earlier mirror)."""
        if full == upstream_full:
            return
        if client.image_exists(upstream_full):
            return
        client.tag(full, upstream_full)

    if client.image_exists(full):
        _alias_for_upstream()
        return True
    log.info("pulling %s (the first download may take several minutes)", full)
    try:
        client.pull(full)
        _alias_for_upstream()
    except ContainerError as exc:
        log.warning("pull failed for %s: %s", full, exc)
        return False
    return True


def _persist_diagnostics_if_ifail(
    res: subprocess.CompletedProcess[str],
    per_task_out: Path,
    permanent_out: Path,
    instance_id: str,
) -> None:
    """Persist eval stdout/stderr/returncode when the upstream eval crashed.

    "Infra-failure" pattern: upstream's workspace/output.json never appeared
    (eval container died before parser.py wrote it). Without this we lose
    every diagnostic. Always write the artifacts under
    <permanent_out>/<iid>/_eval_*.{log,txt} for later classification.
    """
    upstream_output = per_task_out / instance_id / "workspace" / "output.json"
    if upstream_output.exists() and upstream_output.stat().st_size > 0:
        return
    target = permanent_out / instance_id
    target.mkdir(parents=True, exist_ok=True)
    try:
        (target / "_eval_stdout.log").write_text(res.stdout or "")
        (target / "_eval_stderr.log").write_text(res.stderr or "")
        (target / "_eval_returncode.txt").write_text(str(res.returncode))
    except OSError:
        # Eval workspace is root-owned and may have left stale perms.
        # Best-effort; if we can't write we just lose this round's diagnostics.
        pass


def _move_eval_workspace(
    per_task_out: Path, permanent_out: Path, instance_id: str
) -> None:
    task_workspace = per_task_out / instance_id
    target = permanent_out / instance_id
    if task_workspace.exists() and not target.exists():
        try:
            shutil.move(str(task_workspace), str(target))
        except (PermissionError, OSError):
            pass


def _has_port_collision(per_task_out: Path, instance_id: str) -> bool:
    task_workspace = per_task_out / instance_id
    for rel in ("_stdout.log", "_stderr.log", "workspace/stdout.log", "workspace/stderr.log"):
        p = task_workspace / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if "EADDRINUSE" in text or "address already in use" in text:
            return True
    return False


def _eval_one(
    repo_dir: Path,
    instance_id: str,
    patch: str,
    *,
    evaluator_dir: Path | None = None,
    dataset_path: Path | None = None,
    docker_host: str | None = None,
) -> bool | None:
    """Run upstream evaluator on a single task. Returns True/False/None.

    Each invocation gets its own temp `--output_dir` to avoid races on the
    eval_results.json file when multiple workers run in parallel. We move
    the per-task workspace/ subdir back into the permanent location for
    later debugging.
    """
    try:
        eval_repo, eval_script, scripts_dir = _resolve_evaluator_paths(evaluator_dir)
        raw_dataset_path = materialize_dataset_jsonl(dataset_path)
    except (OSError, RuntimeError) as exc:
        log.warning("could not prepare evaluator for %s: %s", instance_id, exc)
        return None
    permanent_out = repo_dir / "score"
    permanent_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sbs-eval-") as tmp:
        tmp_path = Path(tmp)
        patches_path = tmp_path / "patches.json"
        patches_path.write_text(
            json.dumps([{"instance_id": instance_id, "patch": patch}], indent=2)
        )
        per_task_out = tmp_path / "out"
        per_task_out.mkdir()
        cmd = [
            sys.executable,
            str(eval_script),
            "--raw_sample_path",
            str(raw_dataset_path),
            "--patch_path",
            str(patches_path),
            "--output_dir",
            str(per_task_out),
            "--scripts_dir",
            str(scripts_dir),
            "--dockerhub_username",
            "jefzda",
            "--num_workers",
            "1",
            "--use_local_docker",
            "--block_network",
        ]
        env = _evaluator_environment(_podman_docker_host(docker_host))
        res = subprocess.run(
            cmd, cwd=eval_repo, env=env, capture_output=True, text=True
        )
        ev = per_task_out / "eval_results.json"
        upstream_output = per_task_out / instance_id / "workspace" / "output.json"
        _persist_diagnostics_if_ifail(
            res, per_task_out, permanent_out, instance_id
        )
        if _has_port_collision(per_task_out, instance_id):
            log.warning("port collision during eval for %s", instance_id)
            _move_eval_workspace(per_task_out, permanent_out, instance_id)
            return None
        # The upstream evaluator records ``false`` in eval_results.json even
        # when its Docker-compatible client cannot reach Podman. A real test
        # run always writes workspace/output.json, so require that artifact
        # before accepting either a pass or a fail as a benchmark result.
        try:
            has_upstream_output = (
                upstream_output.is_file() and upstream_output.stat().st_size > 0
            )
        except OSError:
            has_upstream_output = False
        if not has_upstream_output:
            log.warning("evaluator produced no test output for %s", instance_id)
            _move_eval_workspace(per_task_out, permanent_out, instance_id)
            return None
        if not ev.exists():
            return None
        try:
            new = json.loads(ev.read_text())
        except json.JSONDecodeError:
            return None
        result = _resolved_score(new.get(instance_id))
        if result is None:
            return None
        # Move per-task workspace/ to permanent location (best-effort).
        _move_eval_workspace(per_task_out, permanent_out, instance_id)
        return result


def get_cached_eval_result(repo_dir: Path, instance_id: str, patch: str) -> bool | None:
    """Return cached pass/fail if eval_results.json already has an entry for
    `instance_id` for the same patch, else None.

    The patch hash guard matters for resume: task reruns can replace
    patch.diff while leaving the old instance_id score behind. Reusing by
    instance_id alone reports the previous patch's result for the new patch.
    """
    ev = repo_dir / "score" / "eval_results.json"
    if not ev.is_file():
        return None
    try:
        existing = json.loads(ev.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    val = existing.get(instance_id)
    if not isinstance(val, bool):
        return None
    meta = repo_dir / "score" / _EVAL_METADATA
    if not meta.is_file():
        return None
    try:
        metadata = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    expected_hash = metadata.get("patch_sha256", {}).get(instance_id)
    if expected_hash != _patch_sha256(patch):
        return None
    return val


def _merge_eval_result(
    repo_dir: Path,
    instance_id: str,
    passed: bool,
    *,
    patch_sha256: str | None = None,
) -> None:
    """Atomically merge a single (iid -> bool) entry into eval_results.json.

    Safe under concurrent writers. INDEP mode runs up to ~40 tasks per repo
    in parallel; each completion calls this. The write must serialize the
    whole read/modify/write cycle, not just use atomic replace. Otherwise two
    workers can both read the same old JSON and the last replace drops the
    other worker's new entry.
    """
    import os
    import uuid

    score_dir = repo_dir / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    ev = score_dir / "eval_results.json"

    with _merge_lock(score_dir):
        if ev.is_file():
            try:
                existing = json.loads(ev.read_text())
            except json.JSONDecodeError:
                existing = {}
        else:
            existing = {}
        existing[instance_id] = bool(passed)
        # Per-call tmp path → no contention on the .tmp filename.
        tmp = score_dir / f"eval_results.json.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(ev)

        if patch_sha256 is None:
            return
        meta = score_dir / _EVAL_METADATA
        if meta.is_file():
            try:
                metadata = json.loads(meta.read_text())
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = {}
        metadata.setdefault("patch_sha256", {})[instance_id] = patch_sha256
        tmp_meta = score_dir / f"{_EVAL_METADATA}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        tmp_meta.replace(meta)


def score_predictions(
    *,
    predictions_path: Path,
    output_dir: Path,
    evaluator_dir: Path,
    dataset_path: Path | None = None,
    num_workers: int = 1,
    docker_host: str | None = None,
    gradeable_instance_ids: set[str] | None = None,
) -> int:
    """Score predictions and retain only evaluator-backed pass/fail results.

    The upstream evaluator can emit ``false`` when its container infrastructure
    fails before running tests. As in :func:`score_one`, a non-empty patch is
    accepted only when that invocation also produced ``workspace/output.json``.
    Empty patches are deterministic benchmark failures and need no container.
    """

    eval_repo, eval_script, scripts_dir = _resolve_evaluator_paths(evaluator_dir)
    raw_dataset_path = materialize_dataset_jsonl(dataset_path)
    predictions = json.loads(predictions_path.read_text())
    if isinstance(predictions, dict):
        patches: list[object] = [
            {"instance_id": instance_id, "patch": patch}
            for instance_id, patch in predictions.items()
        ]
    elif isinstance(predictions, list):
        patches = predictions
    else:
        raise ValueError(
            f"unsupported predictions format: {type(predictions).__name__}"
        )

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(patches):
        if not isinstance(item, dict):
            raise ValueError(f"prediction at index {index} must be an object")
        instance_id = item.get("instance_id")
        patch = item.get("patch")
        if not isinstance(instance_id, str) or not isinstance(patch, str):
            raise ValueError(
                f"prediction at index {index} must contain string instance_id and patch"
            )
        if instance_id in seen_ids:
            raise ValueError(f"duplicate prediction for {instance_id}")
        seen_ids.add(instance_id)
        normalized.append({"instance_id": instance_id, "patch": patch})
    if gradeable_instance_ids is not None:
        normalized = [
            item
            for item in normalized
            if item["instance_id"] in gradeable_instance_ids
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    patches_path = output_dir / "patches.json"
    patches_path.write_text(json.dumps(normalized, indent=2) + "\n")

    results = {
        item["instance_id"]: False
        for item in normalized
        if not item["patch"].strip()
    }
    submitted = [item for item in normalized if item["patch"].strip()]
    returncode = 0
    missing: list[str] = []

    if submitted:
        with tempfile.TemporaryDirectory(prefix="sbs-bulk-eval-") as tmp:
            invocation_dir = Path(tmp)
            submitted_path = invocation_dir / "patches.json"
            submitted_path.write_text(json.dumps(submitted, indent=2) + "\n")
            evaluator_output = invocation_dir / "out"
            evaluator_output.mkdir()
            cmd = [
                sys.executable,
                str(eval_script),
                "--raw_sample_path",
                str(raw_dataset_path),
                "--patch_path",
                str(submitted_path),
                "--output_dir",
                str(evaluator_output),
                "--scripts_dir",
                str(scripts_dir),
                "--dockerhub_username",
                "jefzda",
                "--num_workers",
                str(num_workers),
                "--use_local_docker",
                "--block_network",
            ]
            completed = subprocess.run(
                cmd,
                cwd=eval_repo,
                env=_evaluator_environment(_podman_docker_host(docker_host)),
                check=False,
            )
            returncode = completed.returncode
            try:
                raw_results = json.loads(
                    (evaluator_output / "eval_results.json").read_text()
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                raw_results = {}
            if not isinstance(raw_results, dict):
                raw_results = {}

            for item in submitted:
                instance_id = item["instance_id"]
                upstream_output = (
                    evaluator_output / instance_id / "workspace" / "output.json"
                )
                try:
                    has_output = (
                        upstream_output.is_file()
                        and upstream_output.stat().st_size > 0
                    )
                except OSError:
                    has_output = False
                score = _resolved_score(raw_results.get(instance_id))
                if not has_output or score is None:
                    missing.append(instance_id)
                    log.warning(
                        "bulk evaluator produced no trustworthy result for %s",
                        instance_id,
                    )
                else:
                    results[instance_id] = score
                _replace_eval_workspace(evaluator_output, output_dir, instance_id)

    _write_bulk_results(output_dir, results)
    if returncode != 0:
        return returncode
    return 1 if missing else 0


def _replace_eval_workspace(source_root: Path, target_root: Path, instance_id: str) -> None:
    """Replace one bulk-evaluation workspace with artifacts from this invocation."""

    source = source_root / instance_id
    if not source.exists():
        return
    target = target_root / instance_id
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
    except OSError as exc:
        log.warning("could not persist evaluator artifacts for %s: %s", instance_id, exc)


def _write_bulk_results(output_dir: Path, results: dict[str, bool]) -> None:
    import uuid

    target = output_dir / "eval_results.json"
    temporary = output_dir / f"eval_results.json.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def score_one(
    repo_dir: Path,
    instance_id: str,
    patch: str,
    tag_lookup: dict[str, str] | None = None,
    image_prefix: str = _IMAGE_PREFIX,
    *,
    evaluator_dir: Path | None = None,
    dataset_path: Path | None = None,
    docker_host: str | None = None,
    podman: PodmanCli | None = None,
) -> bool | None:
    """Score one task synchronously. Returns True/False (pass/fail) or None
    (eval-infra failure: image pull failed, evaluator crashed, etc.).

    Side effects:
      - podman pull <image> (if not present)
      - runs upstream evaluator in a temp output_dir
      - on True/False: merges {iid: result} into repo_dir/score/eval_results.json
      - on None: leaves eval_results.json untouched, writes diagnostic
        logs under repo_dir/score/<iid>/_eval_*.{log,txt}

    Empty `patch` (e.g., compaction-thrash failures producing 0-byte
    patch.diff) short-circuits to False without an eval call — the
    evaluator would record FAIL anyway, no point burning podman cycles.

    Pass `tag_lookup=load_tag_lookup(dataset_path)` once and reuse to avoid
    reloading the dataset for every call.
    """
    patch_hash = _patch_sha256(patch)
    if not patch.strip():
        _merge_eval_result(
            repo_dir, instance_id, False, patch_sha256=patch_hash
        )
        return False
    if tag_lookup is None:
        tag_lookup = load_tag_lookup(dataset_path)
    tag = tag_lookup.get(instance_id)
    if not tag:
        log.warning("score_one: no dockerhub_tag for %s", instance_id)
        return None
    client = podman or PodmanCli()
    if not _ensure_image(tag, image_prefix=image_prefix, podman=client):
        return None
    result = _eval_one(
        repo_dir,
        instance_id,
        patch,
        evaluator_dir=evaluator_dir,
        dataset_path=dataset_path,
        docker_host=docker_host,
    )
    if result is not None:
        _merge_eval_result(
            repo_dir, instance_id, result, patch_sha256=patch_hash
        )
    return result
