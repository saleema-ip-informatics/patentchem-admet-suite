# EGFR SageMaker deployment

## Verified deployment
- Region: ap-southeast-2
- Endpoint: egfr-qsar-serverless
- Configuration: egfr-qsar-serverless-v1
- Model: egfr-qsar-sm-v1
- Execution role: EGFR-SageMaker-ExecutionRole
- Serverless memory: 2048 MB
- Maximum concurrency: 1
- Provisioned concurrency: none
- Model run: 20261021T114548349412Z
- Deployed image tag: sagemaker-v2
- Deployed digest: sha256:cb49e0ae999bff5d1cee1db3178019705b8fbd5ddae23d987618459862b422b6

## Verification
Local health check returned HTTP 200.
AWS inference returned pIC50 7.545550809809083 for CCO,
matching the local result within floating-point precision.
Nearest training similarity: 0.13636363636363635.

This verifies software deployment, not biological activity.
The learning model uses 38 molecules; its applicability domain
has not been assessed. Similarity is not calibrated confidence.

## Access
Invoke through the authenticated SageMaker Runtime API.
This endpoint is not a public Swagger website.
Use AWS CLI profile egfr-dev for deployment commands.

## Build
Run from the repository root:
docker build --platform linux/amd64 --provenance=false -f mlops/Dockerfile.sagemaker -t egfr-qsar:rebuild-check .

For SageMaker uploads, export with oci-mediatypes=false,
provenance=false and sbom=false. Use a new immutable ECR tag.

The deployed image predates the self-contained Dockerfile.
The revised source-based image was tested locally but has
not replaced the deployed image.

## Cost and cleanup
Budget email verification is unresolved in AWS Support.
On-demand inference, image storage and logs can consume credits.
Check Billing regularly.

To remove the endpoint when practice is finished:
aws sagemaker delete-endpoint --endpoint-name egfr-qsar-serverless --profile egfr-dev --region ap-southeast-2

Deleting the endpoint leaves its configuration, model definition,
ECR images and CloudWatch logs for separate review and cleanup.
