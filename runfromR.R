# install reticulate (Only first time)
install.packages("reticulate")
# import reticulate
library(reticulate)
# add path to project folder
setwd("/Users/carlopez/Git-repos/mlss")
# Create virtual env
virtualenv_create(
  envname = "mlss-venv",
  version = ">=3.8",
  requirements = "requirements.txt",
  force = FALSE
)
# use created venv
use_virtualenv(virtualenv = "mlss-venv", required = T)
# run main
command <- "python3 main.py data.externals=/Users/carlopez/Git-repos/mlss/data/sample  surveys=gharb_9 survey_version=all"
system(command)

# uncomment to delete venv
# virtualenv_remove("mlss-venv")
