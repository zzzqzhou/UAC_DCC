# Preparing
Put all datasets under ./data.

# Pretrained Backbone:

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing)
```
├── ./pretrained
    ├── resnet50.pth
    └── resnet101.pth
```

# Training Scripts

```bash
# use torch.distributed.launch
sh tools/dist_train.sh <config> <port> <save_path> <num_gpu>
```
```bash
# use slurm
GPUS=<num_gpu> GPUS_PER_NODE=<num_gpu> CPUS_PER_TASK=<num_cpu> sh tools/slurm_train.sh <partition> <config> <port> <save_path>
```
