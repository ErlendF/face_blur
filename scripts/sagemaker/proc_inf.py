import boto3
import os
import sys
from datetime import datetime

# Storing more sensitive variables in separate file to avoid commiting them
import variables

now = datetime.now().strftime("%Y%m%d%H%M%S")

# This really shouldn't be necessary, but it is
os.environ["AWS_PROFILE"] = variables.aws_account
os.environ["AWS_DEFAULT_REGION"] = variables.region


session = boto3.Session(profile_name=variables.aws_account)
smc = boto3.client("sagemaker", region_name=variables.region)
ecrc = boto3.client("ecr", region_name=variables.region)

# Cleaning up untagged images
img_resp = ecrc.list_images(
    repositoryName=variables.repository_name,
    filter={
        "tagStatus": "UNTAGGED"
    }
)

if len(img_resp["imageIds"]) != 0:
    print("Cleaning up {} untagged images".format(len(img_resp["imageIds"])))
    img_del_resp = ecrc.batch_delete_image(
        repositoryName=variables.repository_name,
        imageIds=img_resp["imageIds"]
    )
    if img_del_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
        print("Could not clean up untagged images: {}".format(img_del_resp))

# # Deleting existing models
# models_resp = smc.list_models()
# for model in models_resp["Models"]:
#     print("Removing model {}".format(model["ModelName"]))
#     del_resp = smc.delete_model(ModelName=model["ModelName"])
#     if del_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
#         print("Could not delete model {}".format(model["ModelName"]))

# # Making new model
# model_name = "model-" + now
# print("Making new model {}".format(model_name))

# new_model_resp = smc.create_model(
#     ModelName=model_name,
#     PrimaryContainer={
#         "Image": variables.imageURI,
#         # "ModelDataUrl": variables.model_s3_uri
#     },
#     ExecutionRoleArn=variables.smRoleArn,
#     Tags=variables.tags
# )
# if new_model_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
#     print("Could not make model: {}".format(new_model_resp))
#     sys.exit(1)

# print(f"new model resp: {new_model_resp}")

# Making batch transform job
processing_job_name = "test-processing-" + now
print("Making processing job {}".format(processing_job_name))

proc_resp = smc.create_processing_job(
    ProcessingJobName=processing_job_name,
    ProcessingInputs=[{
        "InputName": processing_job_name + "-input",
        "S3Input": {
            "S3DataType": "S3Prefix",
            "S3Uri": variables.data_input,
            "S3InputMode": "File",
            "LocalPath": variables.data_input_local
        }
    }],
    ProcessingOutputConfig={
        "Outputs": [{
            "OutputName": processing_job_name + "-output",
            "S3Output": {
                "S3Uri": variables.data_output,
                'LocalPath': variables.data_output_local,
                'S3UploadMode': 'Continuous'
            }
        }]
    },
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount":variables.instance_count,
            "InstanceType": variables.instance_type,
            "VolumeSizeInGB": variables.volume_size_in_gb,
        }
    },
    AppSpecification={
        "ImageUri": variables.imageURI
    },
    RoleArn=variables.smRoleArn,
    Tags=variables.tags
)

if proc_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
    print("Could not make batch transform job: {}".format(proc_resp))
    sys.exit(1)

print("Success!")
