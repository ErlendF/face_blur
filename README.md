# Face blur

## Examples
Some simple example configurations are provided in the *examples* folder. 

## Models
Some example models in the *models* folder, both related to this library, Insightface and Arcface/MTCNN. The models are made either for AWS SageMaker Batch Transform Jobs or Processing Job.

These models may be built using:
```
docker build . -t <tag>
```

Afterwards, the docker image may be pushed to ECR. Commands and instructions for how to do so is provided in the ECR repository itself.



## Scripts
Scripts are provided in the *scripts* folder. These scripts may be used to deploy both AWS SageMaker Batch Transform Jobs and Processing Jobs. They require the models to be pushed to ECR, and require some variables to be defined. These variables should be put in a file called *variables.py*, an example of which is provided in the *scripts/ex_variables.py* file.