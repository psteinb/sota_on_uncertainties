# SOTA with uncertainties

trying to obtain uncertainties from training accuracies using [timm](https://github.com/rwightman/pytorch-image-models/).

# Required Environment

## instructions for a vanilla python installation

We assume you have some form of GPU available including the required runtime environment available. If not, you can try to execute the workflow on CPU-only hardware. Note though that should you wish to train the networks, running on a CPU-only hardware can be very slow.

Checking the python version:

```bash
python --version
```
should give:
```
3.8.5
```
Set up the `venv`:
```bash
python -m venv <some-name>
```
Source the `venv`:
```bash
source <some-name>/bin/activate
```

## instructions for Jusuf

For development:

```bash
salloc -N 1 -p gpus -A <omitted> -t 01:00:00
```

```bash
srun --cpu_bind=none --pty /bin/bash -i
```

Setup the software environment:

```bash
ml add Stages/2020 GCCcore/.10.3.0 CUDA/11.3 Python/3.8.5
```
This will setup the environment to:

```bash
ml
```
this produces:
```
Currently Loaded Modules:
  1) Stages/2020            (S)     9) ncurses/.6.2     (H)  17) util-linux/.2.36    (H)  25) libxml2/.2.9.10  (H)  33) libspatialindex/.1.9.3 (H)
  2) StdEnv/2020                   10) libreadline/.8.0 (H)  18) fontconfig/.2.13.92 (H)  26) libxslt/.1.1.34  (H)  34) NASM/.2.15.03          (H)
  3) GCCcore/.10.3.0        (H)    11) Tcl/8.6.10            19) xorg-macros/.1.19.2 (H)  27) libffi/.3.3      (H)  35) libjpeg-turbo/.2.0.5   (H)
  4) binutils/.2.36.1       (H)    12) SQLite/.3.32.3   (H)  20) libpciaccess/.0.16  (H)  28) libyaml/.0.2.5   (H)  36) Python/3.8.5
  5) zlib/.1.2.11           (H)    13) expat/.2.2.9     (H)  21) X11/20200222             29) Java/15.0.1
  6) nvidia-driver/.default (H,g)  14) libpng/.1.6.37   (H)  22) Tk/.8.6.10          (H)  30) PostgreSQL/12.3
  7) CUDA/11.3              (g)    15) freetype/.2.10.1 (H)  23) GMP/6.2.0                31) protobuf/.3.13.0 (H)
  8) bzip2/.1.0.8           (H)    16) gperf/.3.1       (H)  24) XZ/.5.2.5           (H)  32) gflags/.2.2.2    (H)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
   g:  built for GPU
   H:             Hidden Module

```
Checking the python version:

```bash
python --version
```
should give:
```
3.8.5
```
Set up the `venv`:
```bash
python -m venv <some-name>
```
Source the `venv`:
```bash
source <some-name>/bin/activate
```

For the sake of good practice, update `pip` whenever possible:

```bash
python -m pip install -U pip
```


# Required Python Packages

### Prepare for full training

To prepare the environment and set up `timm` for complete training, we need to install our own pytorch:

```bash
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Should you not have any GPU (or CUDA aware) hardware available, note that pytorch can also be installed cpu-only:

```bash
python -m pip install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

Whether with GPU support or not, you need to install the full software stack:

```bash
python -m pip install -r requirements-full.txt
```

**NB** We did not test our workflow in this scenario.

### Prepare for reproducing the figures only

**Note**: If you are not interested to rerun the machine learning traing, you are fine to go without `pytorch` and `timm` such as:

```bash
python -m pip install -r requirements.txt
```


# Running the experiments

## Getting the data

Note, this repo involves 360 1h runs on a Nvidia V100. If you'd like to repeat the experiments, you need to download `imagenette2` the dataset as documented in the timmdocs. Our workflow can do that automatically. Just in case you want to look around first, here is how to obtain it:

```bash
mkdir data
cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xf imagenette2-320.tgz
```

## training all models

To run the 360 experiments sequentially, do

```bash
cd /root/to/repo
```
to switch the current working directory. Then run the training:

```bash
snakemake -j1 imagenette2_train
```

Please use the issue tracker to report any shortcomings.

## parallel execution on a cluster

This workflow setup is prepared with a [slurm](https://slurm.schedm.com) cluster in mind. JUSUF at JSC is managed by [slurm](https://slurm.schedm.com). 

Before you do anything, the destination folders for the slurm logs have to be created.

```bash
cd /path/to/repo
mkdir -p outputs/slurm/logs/
```

On Jusuf, you can run all model trainings as

```bash
snakemake -j40 -p --profile config/slurm/jusuf imagenette2_train
```

Note, this will submit `360` jobs in total, but only run `40` jobs at a time. You can only invoke this command from the `venv` described above. If you'd like to run this on another cluster, adjust `config/slurm/jusuf/config.yaml` to your needs (see [slurm profile](https://github.com/Snakemake-Profiles/slurm) for the api documentation of `config.yaml`).

## Inference (default workflow target)

The default workflow target is to run inference on the validation datasets created. You need at least one GPU for this and all `last.pth.tar` model files generated by `timm` in a folder structure which the workflow expects. In other words:

```bash
snakemake -j80 -p --profile config/slurm/jusuf imagenette2_inference_last
```

# Viewing the execution graph

```bash
snakemake -j1 -F --dag results/figures/imagenette2_compare_meanstd_approx.png| dot -Tsvg > ~/imagenette2_compare_meanstd_approx_dag.svg
```
