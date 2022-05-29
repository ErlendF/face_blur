repository_name = "my-test-ecr-repo"
region = "eu-north-1"
aws_account = "my-aws-account"
imageURI = "uri-of-my-image"
smRoleArn = "my-role-arn"
data_input = "s3://path/to/my/s3/location"
data_output = "s3://path/to/my/s3/location"
data_input_local = "/opt/ml/processing/input/"
data_output_local = "/opt/ml/processing/output/"
instance_count = 1
instance_type = "ml.t3.large"
volume_size_in_gb = 20
tags = [
    {
        "Key": "User",
        "Value": "Me!"
    },
    {
        "Key": "Team",
        "Value": "Us!"
    }
]
