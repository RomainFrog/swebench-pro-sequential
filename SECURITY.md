# Security policy

## Supported versions

Security fixes are provided for the latest tagged release and the default
branch.

## Reporting a vulnerability

Please use GitHub's private **Report a vulnerability** flow for this
repository. If private reporting is unavailable, contact the repository owner
through their GitHub profile without posting exploit details publicly.

Include the affected version, impact, minimal reproduction, and any suspected
credential or artifact exposure. Never attach live API keys, private benchmark
traces, or proprietary repository content.

## Execution boundary

Benchmark images and issue content are untrusted. The harness runs task code in
rootless Podman, but a container boundary is not equivalent to a hardened
multi-tenant sandbox. Run evaluations on a dedicated machine or VM, keep model
credentials on the host, avoid mounting unrelated directories, and review
artifacts before publishing them.
