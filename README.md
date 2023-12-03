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


## Local Setup (Recommended)

Ensure that you are using python 3.11

Run `pip install -r requirements.txt` to install the necessary packages

## Overview

This codebase attempts to replicate some of the results presented in the following paper: https://www.nature.com/articles/s41598-020-58053-z

Here we are measuring the performance of several different deep learning algorithms on the ability to predict patient readmission to the ICU. Patient data extracted from the MIMIC-III dataset is used as input to each of the models. This patient data includes the age, location, ethnicity, marital status, insurance type, and length of stay in the hospital/ICU for a given patient, as well as any prescriptions or procedures they may have been given.

## Data Processing

Create a new directory `./data/mimic-iii-data` and place the following **unzipped** csv files in there:
- ADMISSIONS.csv
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

## Training

Once you have processed the data, run `python ./train.py --model_type <MODEL_TYPE>` to begin training. Possible values for `<MODEL_TYPE>` are `ode_rnn`, `rnn_exp_decay`, `rnn_concat_time_delta`. Add a `-h` flag to view additional training options, such as number of epochs, batch size, learning rate, etc.

Models get saved after every epoch. Existing models are automatically loaded. If you have an existing model saved but want to train from scratch, make sure you delete the corresponding `.pt` file. The default device is CUDA, but if CUDA is unavailable on your machine, it will give a warning and default to CPU.

## Testing

Similar to training, run `python ./test.py --model_type <MODEL_TYPE>` to begin testing a model after training. Add a `-h` flag to view more testing options.