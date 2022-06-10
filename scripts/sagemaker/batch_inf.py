import boto3
import os
import sys
from datetime import datetime

# Storing more sensitive variables in separate file to avoid committing them
import variables

repository_name = "test_sagemaker_models"
region = "eu-north-1"
tags = [
    {
        "Key": "User",
        "Value": "Erlend"
    },
    {
        "Key": "Team",
        "Value": "Platform"
    }
]

now = datetime.now().strftime("%Y%m%d%H%M%S")

# This really shouldn't be necessary, but it is
os.environ["AWS_PROFILE"] = variables.aws_account
os.environ["AWS_DEFAULT_REGION"] = region


session = boto3.Session(profile_name=variables.aws_account)
smc = boto3.client("sagemaker", region_name=region)
ecrc = boto3.client("ecr", region_name=region)

# Cleaning up untagged images
img_resp = ecrc.list_images(
    repositoryName=repository_name,
    filter={
        "tagStatus": "UNTAGGED"
    }
)

if len(img_resp["imageIds"]) != 0:
    print("Cleaning up {} untagged images".format(len(img_resp["imageIds"])))
    img_del_resp = ecrc.batch_delete_image(
        repositoryName=repository_name,
        imageIds=img_resp["imageIds"]
    )
    if img_del_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
        print("Could not clean up untagged images: {}".format(img_del_resp))

# Deleting existing models
models_resp = smc.list_models()
for model in models_resp["Models"]:
    print("Removing model {}".format(model["ModelName"]))
    del_resp = smc.delete_model(ModelName=model["ModelName"])
    if del_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
        print("Could not delete model {}".format(model["ModelName"]))

# Making new model
model_name = "model-" + now
print("Making new model {}".format(model_name))

new_model_resp = smc.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": variables.imageURI,
        # "ModelDataUrl": variables.model_s3_uri
    },
    ExecutionRoleArn=variables.smRoleArn,
    Tags=tags
)
if new_model_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
    print("Could not make model: {}".format(new_model_resp))
    sys.exit(1)

# Making batch transform job
transform_job_name = variables.job_prefix + "test-transform-" + now
print("Making batch transformjob {}".format(transform_job_name))

tf_resp = smc.create_transform_job(
    TransformJobName=transform_job_name,
    ModelName=model_name,
    MaxConcurrentTransforms=variables.max_concurrent_transforms,
    BatchStrategy="MultiRecord",
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": variables.data_input
            }
        },
        "ContentType": "application/x-image",
        "CompressionType": "None",
        "SplitType": "None"
    },
    TransformOutput={
        "S3OutputPath": variables.data_output,
        "AssembleWith": "None"
    },
    TransformResources={
        "InstanceType": variables.instance_type,
        "InstanceCount": variables.instance_count
    },
    Tags=tags
)
if tf_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
    print("Could not make batch transform job: {}".format(tf_resp))
    sys.exit(1)

print("Success!")
