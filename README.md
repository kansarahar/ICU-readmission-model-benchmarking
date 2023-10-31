# ICU Readmission Model Benchmarking


## Docker Setup

To get started with docker:

Run 
```
docker run -it --privileged=true --cap-add=SYS_ADMIN -m 8192m -h torch.local --gpus all --name torch-container -v <<<path/to/your/local/repo>>>:/workspace/ pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel /bin/bash
```
to create the container.

Run `docker exec -it <CONTAINER_ID> bash` anytime after to re-enter the container.

