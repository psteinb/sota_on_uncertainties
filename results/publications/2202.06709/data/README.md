all csv files have been obtained from the figures in [2202.06709](https://arxiv.org/abs/2202.06709) using [WebPlotDigitizer](https://apps.automeris.io/wpd/) on Feb 28, 2022

## fig12

The paper claims to have produced the figures using CIFAR-100-C:

> Robustness is mean accuracy on CIFAR-100-C

For CIFAR-100-C, the paper refers to [Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261v1) which in turn references this dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3555552.svg)](https://doi.org/10.5281/zenodo.3555552).

Whereas the text refers to CIFAR-100 ([available online](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)) for all measurements related to accuracy. CIFAR100 as loaded through `torchvision` exposes:

```
Train: 50000, Test: 10000, Classes: 100
```
