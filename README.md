# ICU Readmission Model Benchmarking

## Resources
Adapted from the original publication codebase: https://github.com/sebbarb/time_aware_attention

## Docker Setup

Docker is not necessary for this project, but to get started with docker:

Run
```
docker run -it --privileged=true --cap-add=SYS_ADMIN -m 8192m -h torch.local --gpus all --name torch-container -v <<<path/to/your/local/repo>>>:/workspace/ pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel /bin/bash
```
to create a container.

Run `docker exec -it <CONTAINER_ID> bash` anytime after to re-enter the container.

Run `apt-get update && apt-get install -y git` to install git on the docker container

Run `pip install torchdiffeq`


## Local Setup

Ensure that you are using python 3.11

Run `pip install -r requirements.txt` to install the necessary packages

## Data Processing

Create a new directory `./data/mimic-iii-data` and place the following **unzipped** csv files in there:
- ADMISSIONS.CSV
- CHARTEVENTS.csv
- D_ICD_DIAGNOSES.csv
- D_ICD_PROCEDURES.csv
- D_ITEMS.csv
- DIAGNOSES_ICD.csv
- ICUSTAYS.csv
- OUTPUTEVENTS.csv
- PATIENTS.csv
- PRESCRIPTIONS.csv
- PROCEDURES_ICD.csv
- SERVICES.csv

Then you can run `python ./etl/etl.py` to process the raw data. This should take roughly 10 minutes to complete. If it fails part way through, you can easily find out where it failed and run the failed process by itself.