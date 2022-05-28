variable "default_tags" {
  type        = map(string)
  description = "Default tags used by the AWS provider"
}

variable "region" {
  type        = string
  description = "AWS region"
}

variable "aws_account_id" {
  type        = string
  description = "AWS account id"
}

variable "notebook_name" {
  type        = string
  description = "Name of the notebook"
}

variable "notebook_instance_type" {
  type        = string
  default     = "ml.t3.medium"
  description = "Instance type used to run the notebook"
}

variable "notebook_volume_size" {
  type        = number
  default     = 5
  description = "Size of the volume connected to the notebook instance in GB"
}

variable "ecr_name" {
  type = string
  default = "test_sagemaker_models"
  description = "Name of the ECR repository"
}
