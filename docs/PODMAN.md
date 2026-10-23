# Podman setup

The harness runs task containers with the rootless Podman CLI. The official
evaluator talks to Podman's Docker-compatible API socket through the `docker`
Python package; it does not require a Docker daemon or Docker CLI. Linux with
rootless Podman is the primary tested environment. Windows is not currently
tested.

## API socket

Linux with systemd:

```bash
systemctl --user enable --now podman.socket
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
```

Linux without systemd—leave the service running in one terminal:

```bash
podman system service --time=0 unix:///tmp/podman.sock
```

Then use this in the evaluation terminal:

```bash
export DOCKER_HOST=unix:///tmp/podman.sock
```

macOS—allocate enough Podman VM disk and memory for the repositories you
select:

```bash
podman machine start
export DOCKER_HOST="unix://$(podman machine inspect \
  --format '{{.ConnectionInfo.PodmanSocket.Path}}')"
```

## Disk usage

Completed task containers are removed automatically. Images stay in Podman's
cache for reuse across runs, and the harness never deletes pre-existing image
tags. A full run may require hundreds of gigabytes of image layers; check
`podman system df` and the free-space line from `swebench-seq doctor` first.
Disk cleanup remains an explicit user operation such as `podman image prune`.

## Proxies, networking, and environment variables

Ordinary `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` variables are honored and
forwarded to task containers. Advanced environment controls are:

- `SWEBENCH_PODMAN_COMMAND`: alternate Podman executable or wrapper;
- `SWEBENCH_ROLLOUT_PROXY`: proxy forwarded into task containers;
- `SWEBENCH_CONTAINER_PROXY`: fallback proxy for container operations;
- `SWEBENCH_ROLLOUT_NETWORK_MODE`: Podman network mode for task containers;
- `SWEBENCH_CONTAINER_NETWORK_MODE`: fallback container network mode;
- `SWEBENCH_PRO_EVALUATOR_DIR`: local evaluator checkout override.

Rollout containers use Podman's default network unless a rollout network mode
is set explicitly. Evaluator containers always run with network access blocked.
Proxy URLs are visible to code inside the rollout container, so avoid embedding
reusable credentials in them.
