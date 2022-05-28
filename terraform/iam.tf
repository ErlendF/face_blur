resource "aws_iam_role" "notebook" {
  name               = "${var.notebook_name}-role"
  assume_role_policy = data.aws_iam_policy_document.notebook-assume-role.json
}

data "aws_iam_policy_document" "notebook-assume-role" {
  statement {
    actions = [
      "sts:AssumeRole"
    ]

    principals {
      identifiers = ["sagemaker.amazonaws.com"]
      type        = "Service"
    }

    effect = "Allow"
  }
}

resource "aws_iam_role_policy_attachment" "notebook-s3" {
  role       = aws_iam_role.notebook.id
  policy_arn = aws_iam_policy.notebook-s3.arn
}

resource "aws_iam_policy" "notebook-s3" {
  name   = "${var.notebook_name}-s3-policy"
  policy = data.aws_iam_policy_document.notebook-s3.json
}

data "aws_iam_policy_document" "notebook-s3" {
  statement {
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload"
    ]
    resources = [
      aws_s3_bucket.test_data.arn,
      "${aws_s3_bucket.test_data.arn}/*",
    ]
  }
}

# CloudWatch logs permissions
resource "aws_iam_role_policy_attachment" "notebook-cw" {
  role       = aws_iam_role.notebook.id
  policy_arn = aws_iam_policy.notebook-cw.arn
}

resource "aws_iam_policy" "notebook-cw" {
  name   = "${var.notebook_name}-cw-policy"
  policy = data.aws_iam_policy_document.notebook-cw.json
}

data "aws_iam_policy_document" "notebook-cw" {
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
      "cloudwatch:ListMetrics",
      "cloudwatch:PutMetricAlarm",
      "cloudwatch:PutMetricData"
    ]
    resources = [
      "arn:aws:logs:${var.region}:${var.aws_account_id}:log-group:/aws/sagemaker",
      "arn:aws:logs:${var.region}:${var.aws_account_id}:log-group:/aws/sagemaker/*",
    ]
  }
}

# ECR logs permissions
resource "aws_iam_role_policy_attachment" "notebook-ecr" {
  role       = aws_iam_role.notebook.id
  policy_arn = aws_iam_policy.notebook-ecr.arn
}

resource "aws_iam_policy" "notebook-ecr" {
  name   = "${var.notebook_name}-ecr-policy"
  policy = data.aws_iam_policy_document.notebook-ecr.json
}

data "aws_iam_policy_document" "notebook-ecr" {
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:DescribeRepositories",
      "ecr:ListImages",
      "ecr:DescribeImages",
      "ecr:BatchGetImage"
    ]
    resources = [
      "*"
    ]
  }
}

# This is intended for development purposes.
# Permissions should be restricted to the minimum possible in a production environment
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.notebook.id
  policy_arn = data.aws_iam_policy.sagemaker_full_access.arn
}

data "aws_iam_policy" "sagemaker_full_access" {
  arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
