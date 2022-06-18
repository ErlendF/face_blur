import boto3
from videoface import get_shot_transitions
from glob import glob
from os import environ
from sys import exit

# Storing more sensitive variables in separate file to avoid committing them
import variables


def upload_folder_by_shot(dir, dest):
    environ["AWS_PROFILE"] = variables.aws_account
    environ["AWS_DEFAULT_REGION"] = variables.region
    s3_client = boto3.client('s3')

    shots = get_shot_transitions(dir)
    shots = [s for s in shots]

    dest = dest.removesuffix("/")
    current = 0

    for i, filepath in enumerate(sorted(glob(dir))):
        if i > shots[current]:
            current += 1

        filename = filepath.removeprefix(dir).removeprefix("/")
        resp = s3_client.upload_file(
            filepath, dest + "/" + current + "/" + filename)
        if resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
            print("Could not upload file to s3: {}".format(resp))
            exit(1)
