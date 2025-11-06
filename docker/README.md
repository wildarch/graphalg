# GraphAlg Docker Image
The `Dockerfile` in this directory creates a docker image with all dependencies
necessary for development. It is also used as a base image for the devcontainer.

## Building
You only need to do this after updating the `Dockerfile`.

```bash
docker build -t ghcr.io/wildarch/graphalg docker/
docker push ghcr.io/wildarch/graphalg
```
