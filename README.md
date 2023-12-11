<div align="center">
<h1>Zeus<sup>2</sup>: Implementing Structure for Heterogenous GPUs for DNN
Training Energy Savings</h1>
</div>

Zeus is a framework for (1) measuring GPU energy consumption and (2) optimizing energy and time for DNN training. A summary can be found [here](https://ml.energy/zeus/overview/), and the research paper for Zeus can be found [here](https://www.usenix.org/conference/nsdi23/presentation/you). 

Zeus assumes that the same types of GPUs are used during DNN training. This extension accounts for heterogeneous GPUs used in DNN training. First, each GPU is profiled when trained on different datasets so we can plot its power limits and batch sizes. This profiling is used to generate an optimal allocation of both global batch size and power limits across heterogeneous GPUs. Then, for all future cases of DNN training, this optimal allocation can be used.

### Launching GPUs on AWS

We use AWS to simulate heterogeneous GPUs:
1. Launch 2 different EC2 instances. We used g5 and g4dn instances, which correspond to Nvidia A10 and Nvidia T4 GPUs.
2. Select a Deep Learning OSS Nvidia Driver. We selected AMI GPU PyTorch 2.1.0 (Ubuntu 20.04).

### Setting up Zeus

Run a docker container to set up Zeus. Some datasets will also require adding a volume to the run container command.

```
docker run -it \
    --gpus all                  `# Mount all GPUs` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --ipc host                  `# PyTorch DataLoader workers need enough shm` \
    mlenergy/zeus:latest \
    bash
```
* If using Cifar100, setup is complete.
* If using Imagenet, download the [Imagenet](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) dataset and add it as a volume to the docker container.

### Generating Profiling

To generate profiling traces for each GPU, run the following command for the Cifar100 dataset:
```
python zeus/examples/ZeusDataLoader/cifar100/run_profiling.py \
    --profile_folder NAME \
    --epochs 1 \
    --batch_sizes 32 64 128 256 512 1024 \
    --power_limits 70 65 60
```

or the following command for the Imagenet dataset:

The `profile_folder` should be a unique string, `epochs` can be set to 1, `batch_sizes` depend on the dataset, and `power_limits` depend on the GPU type. If needed, `warmup_step` and `profiling_steps` can also be edited via command-line arguments.

The example trace files generated (for Cifar100 and Imagenet on A10 and T4 GPUs) can be viewed in the [`trace_aws`](trace_aws) folder.

### Running Algorithm

### Training Dataset

