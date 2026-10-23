"""Live lifecycle check for the public rootless Podman boundary."""

from __future__ import annotations

import os
import uuid

import docker  # type: ignore[import-untyped]
import pytest

from swebench_seq.evaluation.runner import _docker_api_reports_podman, _podman_docker_host
from swebench_seq.runner import PodmanCli

pytestmark = [
    pytest.mark.podman,
    pytest.mark.skipif(
        os.environ.get("SWEBENCH_RUN_PODMAN_TESTS") != "1",
        reason="set SWEBENCH_RUN_PODMAN_TESTS=1 to run the live Podman test",
    ),
]


def test_rootless_podman_cli_and_compatible_api() -> None:
    podman = PodmanCli()
    image = os.environ.get("SWEBENCH_TEST_IMAGE", "docker.io/library/alpine:3.20")
    name = f"swebench-seq-ci-{uuid.uuid4().hex[:8]}"
    if not podman.image_exists(image):
        podman.pull(image)

    try:
        podman.run_detached(image=image, name=name, env={})
        result = podman.exec(name, cmd=["sh", "-c", "printf ready"], workdir="/")
        identity = podman.image_identity(image)
        assert result.stdout == "ready"
        assert identity["id"]
    finally:
        podman.stop(name, timeout_seconds=1)

    client = docker.APIClient(
        base_url=_podman_docker_host(os.environ.get("DOCKER_HOST")),
        timeout=5,
    )
    try:
        assert client.ping()
        assert _docker_api_reports_podman(client.version())
    finally:
        client.close()
