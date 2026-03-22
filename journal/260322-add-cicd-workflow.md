# Add CI/CD GitHub Actions Workflow

Added `.github/workflows/docker-publish.yml` to automatically build and push the
Docker image to `ghcr.io/johnmathews/documentation-mcp-server` on every push to
`main`.

## Details

- Uses `docker/build-push-action@v6` with `docker/metadata-action@v5` for tagging
- Tags: `latest` (on default branch) and `sha-<short>` for each commit
- Authenticates with `GITHUB_TOKEN` -- no extra secrets needed
- Permissions scoped to `contents: read` and `packages: write`

## Why

The project has a Dockerfile but had no automated build pipeline. This ensures
the container image stays up to date with every merge to main and is available
from `ghcr.io` for deployment.
