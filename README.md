<div align="center">
<h1>Zeus<sup>2</sup>: Implementing Structure for Heterogenous GPUs for DNN
Training Energy Savings</h1>
</div>

Zeus is a framework for (1) measuring GPU energy consumption and (2) optimizing energy and time for DNN training. A summary can be found [here](https://ml.energy/zeus/overview/), and the research paper for Zeus can be found [here](https://www.usenix.org/conference/nsdi23/presentation/you). 

## Getting Started

Refer to [Getting started](https://ml.energy/zeus/getting_started) for complete instructions on environment setup, installation, and integration.

### Docker image

We provide a Docker image fully equipped with all dependencies and environments.
The only command you need is:

```sh
docker run -it \
    --gpus all                  `# Mount all GPUs` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --ipc host                  `# PyTorch DataLoader workers need enough shm` \
    mlenergy/zeus:latest \
    bash
```

Refer to [Environment setup](https://ml.energy/zeus/getting_started/environment/) for details.
