# Face blur

This project is a part of the master's thesis *Automatic blurring of specific faces in video*. The full thesis is publicly available at [the University of Bergen's Open Research Archive](https://bora.uib.no/bora-xmlui/handle/11250/3013662). The following is the abstract of the thesis:

> With the introduction of the General Data Protection Regulation (GDPR) into European Union law, it became more important than ever before to properly handle personal data. This is an issue for media companies which distribute large amounts of media containing identifiable people, which thus may require the subjects' permission for distribution. In this Master's thesis, I propose a solution which supports and facilitates compliance with GDPR regarding the distribution of video containing identifiable subjects by automatically blurring a select group of people in the videos. The proposed solution is a pipeline for detecting, identifying and blurring select faces, where the video frames are processed like individual images to detect and recognize faces, and the interrelatedness of adjacent frames in continuous videos is exploited to both to improve their prediction quality and running time. Each part of the pipeline is interchangeable and may be replaced individually, and the deployment of the entire pipeline has been automated. Aspects related to video processing, facial detection and facial recognition were explored for this purpose, and various existing tools and solutions were utilized.

## Examples
Some simple example configurations are provided in the *examples* folder. More detailed explanations are given in the thesis itself.

## Models
Some example models in the *models* folder, both related to this library, Insightface and Arcface/MTCNN. The models are made either for AWS SageMaker Batch Transform Jobs or Processing Job.

These models may be built using:
```
docker build . -t <tag>
```

Afterwards, the docker image may be pushed to ECR. Commands and instructions for how to do so is provided in the ECR repository itself.



## Scripts
Scripts are provided in the *scripts* folder. These scripts may be used to deploy both AWS SageMaker Batch Transform Jobs and Processing Jobs. They require the models to be pushed to ECR, and require some variables to be defined. These variables should be put in a file called *variables.py*, an example of which is provided in the *scripts/ex_variables.py* file.
