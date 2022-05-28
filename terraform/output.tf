output "s3_bucket_arn" {
  value       = aws_s3_bucket.test_data.arn
  description = "ARN of the S3 bucket, used to gain access to it specifically"
}
