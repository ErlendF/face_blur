# Based on: https://docs.aws.amazon.com/rekognition/latest/dg/create-collection-procedure.html and https://docs.aws.amazon.com/rekognition/latest/dg/add-faces-to-collection-procedure.html
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import os

import variables



def create_collection(client, collection_id):

    client = boto3.client('rekognition')

    # Create a collection
    print('Creating collection:' + collection_id)
    response = client.create_collection(CollectionId=collection_id)
    print('Collection ARN: ' + response['CollectionArn'])
    print('Status code: ' + str(response['StatusCode']))
    print('Done...')


def add_faces_to_collection(client, bucket, photo, collection_id):

    response = client.index_faces(CollectionId=collection_id,
                                  Image={'S3Object': {
                                      'Bucket': bucket, 'Name': photo}},
                                  ExternalImageId=photo,
                                  MaxFaces=1,
                                  QualityFilter="AUTO",
                                  DetectionAttributes=['ALL'])

    print('Results for ' + photo)
    print('Faces indexed:')
    for faceRecord in response['FaceRecords']:
        print('  Face ID: ' + faceRecord['Face']['FaceId'])
        print('  Location: {}'.format(faceRecord['Face']['BoundingBox']))

    print('Faces not indexed:')
    for unindexedFace in response['UnindexedFaces']:
        print(' Location: {}'.format(
            unindexedFace['FaceDetail']['BoundingBox']))
        print(' Reasons:')
        for reason in unindexedFace['Reasons']:
            print('   ' + reason)
    return len(response['FaceRecords'])


def main():
    os.environ["AWS_PROFILE"] = variables.aws_account
    os.environ["AWS_DEFAULT_REGION"] = variables.region


    session = boto3.Session(profile_name=variables.aws_account)
    client = boto3.client('rekognition', region_name=variables.region)

    create_collection(client, variables.collection_id)
    for photo in variables.photos:
        indexed_faces_count = add_faces_to_collection(
            client,
            variables.col_bucket,
            photo,
            variables.collection_id
        )
        print("Faces indexed count: " + str(indexed_faces_count))


if __name__ == "__main__":
    main()
