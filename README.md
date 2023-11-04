# ICU Readmission Model Benchmarking

## Resources
Adapted from the original publication codebase: https://github.com/sebbarb/time_aware_attention

## Docker Setup

To get started with docker:

Run 
```
docker run -it --privileged=true --cap-add=SYS_ADMIN -m 8192m -h torch.local --gpus all --name torch-container -v <<<path/to/your/local/repo>>>:/workspace/ pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel /bin/bash
```
to create the container.

Run `docker exec -it <CONTAINER_ID> bash` anytime after to re-enter the container.

Run `apt-get update && apt-get install -y git` to install git on the docker container

Run `pip install torchdiffeq`