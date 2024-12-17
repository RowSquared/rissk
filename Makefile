#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rissk
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Extract SURVEY value from env.yaml
SURVEY := $(shell $(PYTHON_INTERPRETER) -c "import yaml; print(yaml.safe_load(open('env.yaml'))['SURVEY'])")


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	
	
	R -e "IRkernel::installspec(user = TRUE)"


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 rissk
	isort --check --diff --profile black rissk
	black --check --config pyproject.toml rissk

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml rissk


## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://surveytool/$(SURVEY)/latest/ \
		data/$(SURVEY) \
		--exclude "*" \
		--include "*.zip"

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/$(SURVEY) \
		s3://surveytool/$(SURVEY)/latest \
		--exclude "*.m4a" \
		--exclude "10_RAW/*" \
		--include "10_RAW/**/document.json"

		
	

## Set up python (R) interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) rissk/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
