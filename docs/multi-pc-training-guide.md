# Multi-PC Training Guide

This guide explains how to connect `3` PCs with `RTX 5060` GPUs for one distributed training job.

## Important Current Limitation

The current training code in `src/nuris_pipeline/training/trainer.py` is single-device only.

That means:

- you cannot run true multi-PC training with the repo as it is now
- this guide covers the machine setup and launch pattern you will use after adding distributed training support

## Goal

Use `3` separate PCs as `3` training workers for the same model run.

Typical stack:

- `PyTorch DistributedDataParallel`
- `torchrun`
- one process per GPU
- one GPU per PC

## What You Need

### Hardware

- `3` PCs, each with one working NVIDIA GPU
- enough local disk space for the dataset on every machine
- a stable local network
- wired Ethernet is strongly preferred over Wi-Fi

### Software

Install the same environment on all `3` machines:

- same Python version
- same PyTorch version
- same CUDA-compatible build of PyTorch
- same repo commit
- same `requirements.txt` dependencies

### Data

Each machine should have the same training data at the same path if possible:

- `data/landcover_ai/`
- `data/landcover_ai_patches/`
- `data/landcover_ai_patches/manifest.json`

Using identical paths across machines reduces configuration mistakes.

## How The PCs Connect

Distributed training does not merge the PCs into one Windows box. Instead:

- one machine is the `master`
- the other machines are `workers`
- all machines join the same training session over the network

The master coordinates process startup and gradient synchronization.

## Step 1: Put All PCs On The Same Network

- connect all `3` PCs to the same router or switch
- prefer wired Ethernet
- confirm each machine can reach the others by IP

On Windows PowerShell, find a machine IP:

```powershell
ipconfig
```

Pick one machine as the master. Example:

- `PC1` master: `192.168.1.10`
- `PC2` worker: `192.168.1.11`
- `PC3` worker: `192.168.1.12`

Test connectivity from each machine:

```powershell
ping 192.168.1.10
ping 192.168.1.11
ping 192.168.1.12
```

## Step 2: Use The Same Repo And Environment Everywhere

On each machine:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
```

Confirm CUDA works:

```powershell
py -3.11 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

## Step 3: Copy The Dataset To Every PC

Each PC should have:

- the same patch files
- the same manifest
- the same config

For this repo, that means at minimum:

- `data/landcover_ai_patches/images/`
- `data/landcover_ai_patches/masks/`
- `data/landcover_ai_patches/manifest.json`
- `configs/landcover_ai_training.yaml`

Avoid training over a slow shared network folder if possible. Local SSD copies are better.

## Step 4: Open The Firewall Port

`torchrun` needs a TCP port that all machines can reach.

Example port:

- `29500`

On the master machine, allow inbound traffic for the chosen port if Windows Firewall blocks it.

## Step 5: Add Distributed Training Support To The Code

Before multi-PC training will work, the trainer needs these changes:

- initialize `torch.distributed`
- assign one process to one GPU
- wrap the model in `DistributedDataParallel`
- use `DistributedSampler` for the datasets
- call `sampler.set_epoch(epoch)` during training
- write checkpoints only from rank `0`
- aggregate metrics correctly across ranks

Without this, launching across multiple machines will not train one shared model.

## Step 6: Define Master Address And Rank Values

Each machine needs:

- `MASTER_ADDR`: IP of the master machine
- `MASTER_PORT`: shared port, for example `29500`
- `WORLD_SIZE`: total number of processes, here `3`
- `RANK`: unique global rank for each machine

Example:

- `PC1`: `RANK=0`
- `PC2`: `RANK=1`
- `PC3`: `RANK=2`

Because each PC has one GPU, local GPU index will usually be `0` on each machine.

## Step 7: Launch Training

Example launch command for `PC1`:

```powershell
$env:MASTER_ADDR="192.168.1.10"
$env:MASTER_PORT="29500"
$env:WORLD_SIZE="3"
$env:RANK="0"
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=192.168.1.10 --master_port=29500 -m nuris_pipeline.cli train-segmentation --config configs/landcover_ai_training.yaml
```

Example launch command for `PC2`:

```powershell
$env:MASTER_ADDR="192.168.1.10"
$env:MASTER_PORT="29500"
$env:WORLD_SIZE="3"
$env:RANK="1"
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr=192.168.1.10 --master_port=29500 -m nuris_pipeline.cli train-segmentation --config configs/landcover_ai_training.yaml
```

Example launch command for `PC3`:

```powershell
$env:MASTER_ADDR="192.168.1.10"
$env:MASTER_PORT="29500"
$env:WORLD_SIZE="3"
$env:RANK="2"
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr=192.168.1.10 --master_port=29500 -m nuris_pipeline.cli train-segmentation --config configs/landcover_ai_training.yaml
```

Start the master and workers within a short time window so they can join the same job.

## Recommended Practical Setup

For this repo, the most practical setup is:

- one local copy of the repo on each PC
- one local copy of the dataset on each PC
- same absolute project path if possible
- one machine designated as the master
- wired Ethernet for all machines

## What Performance To Expect

You should not expect perfect `3x` speedup.

For this project, a realistic range is closer to:

- about `2.2x` to `2.7x` if the distributed code is implemented correctly

If one-PC training takes about `11` to `13` hours, `3` PCs may reduce that to roughly:

- about `4` to `6` hours

## Common Problems

### Workers cannot connect

Check:

- wrong `MASTER_ADDR`
- blocked firewall port
- machines on different networks
- `torchrun` launched with mismatched node counts or ranks

### Training hangs at startup

Check:

- one node did not start
- one node has different environment or package versions
- one node cannot read the dataset

### Bad scaling

Check:

- Wi-Fi instead of Ethernet
- CPU dataloader bottlenecks
- disk bottlenecks
- batch size too small

## Recommendation For This Repo

Before building multi-PC support, first try:

- larger `batch_size`
- higher `num_workers`
- a faster single GPU

That is the simpler path.

If you still want multi-PC training, the next engineering step is to implement `DistributedDataParallel` in the current trainer.
