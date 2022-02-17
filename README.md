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
```
This will setup the environment to:
```
$ ml

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
Checking the python version and setting up a `venv`:
```
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


