$ErrorActionPreference = "Stop"

docker build -f docker/Dockerfile -t nuris-v1 .
docker run --rm -v "${PWD}:/app" nuris-v1 run-inference --config configs/v1_inference.yaml
