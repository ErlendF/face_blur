import boto3
import os

import variables


def main():
    os.environ["AWS_PROFILE"] = variables.aws_account
    os.environ["AWS_DEFAULT_REGION"] = variables.region


    session = boto3.Session(profile_name=variables.aws_account)
    client = boto3.client('rekognition', region_name=variables.region)
    response = client.start_face_search(Video={'S3Object': {
                                        'Bucket': variables.video_bucket, 'Name': variables.video}}, CollectionId=variables.collection_id)
    print(response)


if __name__ == "__main__":
    main()
