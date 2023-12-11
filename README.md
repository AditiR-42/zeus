<div align="center">
<h1>Zeus<sup>2</sup>: Implementing Structure for Heterogenous GPUs for DNN
Training Energy Savings</h1>
</div>

Zeus is a framework for (1) measuring GPU energy consumption and (2) optimizing energy and time for DNN training. A summary can be found [here](https://ml.energy/zeus/overview/), and the research paper for Zeus can be found [here](https://www.usenix.org/conference/nsdi23/presentation/you). 

Zeus assumes 

### Launching GPUs on AWS

Refer to [Getting started](https://ml.energy/zeus/getting_started) for complete instructions on environment setup, installation, and integration.

### Setting up Zeus

Run a docker container to set up Zeus. Some datasets will also require adding a volume to the run container command.

```sh
docker run -it \
    --gpus all                  `# Mount all GPUs` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --ipc host                  `# PyTorch DataLoader workers need enough shm` \
    mlenergy/zeus:latest \
    bash
```
* If using Cifar100, setup is complete.
* If using Imagenet, download the [Imagenet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) dataset and add it as a volume to the docker container.
* If using Sentiment140, download the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140/data) dataset and add it as a volume to the docker container.

### Generating Profiling

### Running Algorithm

### Training Dataset

