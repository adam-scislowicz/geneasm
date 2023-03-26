default: all

all: build-docker-image

build-docker-image:sagemaker/docker/Dockerfile sagemaker/docker*
	docker buildx build -t latest -f sagemaker/docker/Dockerfile .

.PHONY: all build-docker-image