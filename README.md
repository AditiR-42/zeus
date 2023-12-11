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

Inside the docker container, clone this repository. Replace any [`Files of Interest`](https://github.com/AditiR-42/zeus_extension/blob/master/README.md#files-of-interest) inside `zeus` with the files from `zeus_extension`.
```
git clone https://github.com/AditiR-42/zeus_extension.git
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
```
python zeus/examples/imagenet/run_profiling.py \
    --profile_folder NAME \
    --epochs 1 \
    --batch_sizes 32 64 128 256 512 \
    --power_limits 70 65 60
```

The `profile_folder` should be a unique string, `epochs` can be set to 1, `batch_sizes` depend on the dataset, and `power_limits` depend on the GPU type. If needed, `warmup_step` and `profiling_steps` can also be edited via command-line arguments. For more information on setting power limits and batch sizes, see [`Determining Constants`](https://github.com/AditiR-42/zeus_extension/blob/master/README.md#determining-constants).

The example trace files generated (for Cifar100 and Imagenet on A10 and T4 GPUs) can be viewed in the [`trace_aws`](trace_aws) folder.

### Running Algorithm

Determine which of the two GPUs is stronger using peta-flop characteristics from their datasheets. In our case, A10 is stronger than T4. Then run the following code, ensuring that gpu1 and trace1 correspond to the stronger of the two GPUs.

```
python zeus_heterogeneous_algorithm.py --gpu1 NAME --gpu2 NAME --trace1 PATH --trace2 PATH
```

The resulting output will show the optimal power limit and global batch size allocation for each GPU using the brute force, heuristic, and baseline methods.

### Training Model

To train the model, run the following command for the Cifar100 dataset:
```
python examples/ZeusDataLoader/cifar100/train.py \
    --epochs INT \
    --power_limit INT \
    --gpu_index INT \ 
    --gpu_split INT 
```

or the following command for the Imagenet dataset:
```
python examples/imagenet/train_single.py \
    --epochs INT \
    --power_limit INT \
    --gpu_index INT \ 
    --gpu_split INT \
    --data /imagenet
```
The `epochs` are user-defined (or can be the default). The `power_limit` should be the optimal power limit obtained from the algorithm output in the previous step. The `gpu_split` is determined by calculating how much of the global batch size is allocated to the stronger GPU. The `gpu_index` should be 0 for the GPU with the smaller workload and 1 for the GPU with the larger workload. For example, a 40-60 workload split would mean `gpu_split` is 40 for both GPUs and `gpu_index` is 0 and 1 for the respective GPUs.

The train files will automatically shard the model across the two GPUs according to the `gpu_split`. The final output of the train files will be the `Time (s)` taken and `Energy (J)` consumed during training. These results can then be compared across the baseline and heuristic methods. 

### Appendix

#### Determining Constants

#### Files of Interest
- [`cifar100/train.py`](examples/ZeusDataLoader/cifar100/train.py)
- [`cifar100/run_profiling.py`](examples/ZeusDataLoader/cifar100/profiling.py)
- [`imagenet/train_single.py`](examples/imagenet/train_single.py)
- [`imagenet/run_profiling.py`](examples/imagenet/run_profiling.py)
- [`optimizer/power_limit.py`](zeus/optimizer/power_limit.py)
- [`zeus_heterogeneous_detailed.ipynb`](zeus_heterogeneous_detailed.ipynb)
- [`zeus_heterogeneous_algorithm.py`](zeus_heterogeneous_algorithm.py)
- [`trace_aws`](trace_aws)
