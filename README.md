<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/img/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/img/logo_light.svg">
  <img alt="Zeus logo" width="55%">
</picture>
<h1>An Energy Optimization Framework for DNN Training</h1>
</div>

Zeus automatically optimizes the **energy and time** of training a DNN to a target validation metric by finding the optimal **batch size** and **GPU power limit**.

Please refer to our [NSDI '23 publication](/) for details.

## Repository Organization

```
.
├── zeus/                # ⚡ Zeus Python package
│   ├── run/             #    - Tools for running Zeus on real training jobs
│   ├── policy/          #    - Optimization policies and extension interfaces
│   ├── profile/         #    - Tools for profiling energy and time
│   ├── simulate.py      #    - Tools for trace-driven simulation
│   ├── util/            #    - Utility functions and classes
│   ├── analyze.py       #    - Analysis functions for power logs
│   └── job.py           #    - Class for job specification
│
├── zeus_monitor/        # 🔌 GPU power monitor
│   ├── zemo/            #    -  A header-only library for querying NVML
│   └── main.cpp         #    -  Source code of the power monitor
│
├── examples/            # 🛠️ Examples of integrating Zeus
│   ├── capriccio/       #    - Integrating with Huggingface and Capriccio
│   ├── cifar100/        #    - Integrating with torchvision and CIFAR100
│   └── trace_driven/    #    - Using the Zeus trace-driven simulator
│
├── capriccio/           # 🌊 A drifting sentiment analysis dataset
│
└── trace/               # 🗃️ Train and power traces for various GPUs and DNNs
```

## Getting Started

Refer to [Getting started](https://ml.energy/zeus/getting_started) for complete instructions on environment setup, installation, and integration.

### Docker image

We provide a Docker image fully equipped with all dependencies and environments.
The only command you need is:

```sh
docker run -it \
    --gpus 1                    `# Mount one GPU` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --shm-size 64G              `# PyTorch DataLoader workers need enough shm` \
    symbioticlab/zeus:latest \
    bash
```

Refer to [Environment setup](https://ml.energy/zeus/getting_started/environment) for details.

### Examples

We provide working examples for integrating and running Zeus:

- [Integrating Zeus with computer vision](examples/cifar100)
- [Integrating Zeus with NLP](examples/capriccio)
- [Running trace-driven simulation on single recurring jobs and the Alibaba GPU cluster trace](examples/trace_driven)


## Extending Zeus

You can easily implement custom policies for batch size and power limit optimization and plug it into Zeus.

Refer to [`docs/extend.md`](docs/extend.md) for details.
