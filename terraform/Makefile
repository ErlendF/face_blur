init:
	terraform init

plan:
	terraform plan -var-file=variables.tfvars -out=tfplan

apply:
	terraform apply "tfplan" && rm tfplan

destroy:
	terraform destroy -var-file=variables.tfvars