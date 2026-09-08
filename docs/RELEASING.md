# Releasing iobrx

The first binary target is CPython 3.11 on Linux x86-64, including WSL2.
The iobrx wheel uses a manylinux2014 baseline; the complete dependency stack
must also support the user's distribution. Ubuntu 24.04 and the Debian 12
container are release test environments. No AVX-512 compilation flags are used.

## One-time publisher setup

In the verified PyPI account, add a pending GitHub Trusted Publisher at
<https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| PyPI project name | `iobrx` |
| GitHub owner | `LCGaoZzz` |
| Repository | `iobrx` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Create the GitHub environment `pypi`. Only tag events publish: the workflow
checks that the tag matches the Python/Rust version and that the commit is
already on `main`. Pull requests and manual runs build and test without
publishing. PyPI uses short-lived OIDC credentials; no PyPI API token belongs
in repository secrets or local files.

GitHub Actions publishes `ghcr.io/lcgaozzz/iobrx:<version>` with its own
`GITHUB_TOKEN`. After the first push, change the container package visibility
to **Public** in its GitHub package settings and verify an anonymous pull.
New GHCR packages are private by default, even for a public source repository.

## Release procedure

1. Update the Python/Rust versions, Dockerfile version, release notes and
   installation examples together. Regenerate `requirements-container.lock`
   with `uv pip compile pyproject.toml --python-version 3.11 --generate-hashes
   --output-file requirements-container.lock` when dependencies change.
2. Open a PR. Both CI and Release must pass. Release builds an x86-64
   manylinux2014 wheel, installs it and all dependencies with
   `--only-binary=:all:` in a new environment, checks all 34 numerical/API/
   portability tests, disabled-native and baseline NumPy dispatch, rebuilds
   the sdist, and tests the final container. No repository `src` path is added
   to the release test environment.
3. Review and merge to `main`, then create and push the matching `v<version>`
   tag. The tag workflow repeats all release gates, pushes the tested
   container, publishes the exact tested wheel/sdist to PyPI, then attaches
   those artifacts, SHA-256 checksums and the container digest to GitHub Release.
4. Verify PyPI metadata and a fresh `pip install --only-binary=:all:
   iobrx==<version>`. Check `pip check`, `iobrx.backend_info()` and analysis
   smoke tests. Verify the public container by its recorded digest.
5. Check Tsinghua's index after it synchronizes. Upload only to PyPI: mirrors
   synchronize independently, and their end-to-end delay is not guaranteed.

Do not reuse a published version for different files. If a publication job
fails, inspect which services already accepted assets before retrying. Fix
code or dependency issues in a new version. A fixed container tag is a
convenience; the release's `container-digest.txt` identifies the immutable image.

The container includes the executed tutorials and public example data at
`/opt/iobrx/tutorials`. It has the analysis/plotting dependencies, but does not
start or include a Jupyter server. It runs as UID 1000 by default; for bind
mounts on Linux, `--user "$(id -u):$(id -g)"` preserves host ownership.
