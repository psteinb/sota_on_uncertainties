# sota_on_uncertainties

trying to obtain uncertainties from training accuracies using timm

## instructions for Jusuf

For development:
```
$ salloc -N 1 -p gpus -A <omitted> -t 01:00:00
$ srun --cpu_bind=none --pty /bin/bash -i
```

Setup the software environment:

```
$ ml add Stages/2020 GCCcore/.10.3.0
$ ml add CUDA/11.3 Python/3.8.5
$ python --version
3.8.5
$ python -m venv <some-name>
$ source <some-name>/bin/activate
```

To prepare the environment and set up timm, we need to install our own pytorch:

```
$ python -m pip torch==1.10.2+cu113 torchvision==0.11.3+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ python -m pip install -r requirements.txt
```


