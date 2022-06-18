import boto3
import os

import variables


def main():
    os.environ["AWS_PROFILE"] = variables.aws_account
    os.environ["AWS_DEFAULT_REGION"] = variables.region


    session = boto3.Session(profile_name=variables.aws_account)
    client = boto3.client('rekognition', region_name=variables.region)
    
    response = client.get_face_search(JobId=variables.job_id)
    print(response)


if __name__ == "__main__":
    main()
