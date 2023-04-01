default: stack-status

stack-status:

stack-up: dashboard_stack.yaml
	aws cloudformation create-stack --stack-name ObservabilityStack --template-body file://dashboard_stack.yaml --capabilities CAPABILITY_NAMED_IAM

stack-down: dashboard_stack.yaml
	aws cloudformation delete-stack --stack-name ObservabilityStack


build-docker-image:sagemaker/docker/Dockerfile sagemaker/docker*
	docker buildx build -t latest -f sagemaker/docker/Dockerfile .

.PHONY: all build-docker-image