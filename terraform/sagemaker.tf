resource "aws_sagemaker_notebook_instance" "notebook" {
  name                = var.notebook_name
  instance_type       = var.notebook_instance_type
  role_arn            = aws_iam_role.notebook.arn
  platform_identifier = "notebook-al2-v1"
  volume_size         = var.notebook_volume_size

  lifecycle_config_name = local.lifecycle_config_name
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook" {
  name      = local.lifecycle_config_name
  on_create = data.local_file.on_create.content_base64
}
