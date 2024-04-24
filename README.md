# GeoFlood

Flood mapping program based on high-resolution terrain analyses.

## Documentation

Under construction

## Usage

See the `examples` directory for Jupyter notebooks that demonstrate how to use `pygeoflood`.

## Installation
`pygeoflood` is not yet available on PyPI. To install from GitHub, use the following command (Python 3.10 or 3.11 is required):

```bash
$ pip install git+https://github.com/passah2o/pygeoflood
```

We recommend installing `pygeoflood` into a virtual environment. Below are commands for installing `pygeoflood` into a `conda` environment[^1].

```bash
$ conda create -n pygeoflood-env python=3.11
$ conda activate pygeoflood-env
(pygeoflood-env) $ pip install git+https://github.com/passah2o/pygeoflood
```

To obtain an editable installation for local development:

```bash
$ conda create -n pygeoflood-env python=3.11
$ conda activate pygeoflood-env
(pygeoflood-env) $ git clone https://github.com/passah2o/pygeoflood
(pygeoflood-env) $ cd geoflood
(pygeoflood-env) $ pip install -e .
```

[^1]: Instructions for installing `conda`: https://docs.anaconda.com/free/miniconda/#quick-command-line-install.

## Citing

When using GeoFlood, please cite the following paper:

Zheng, X., D. Maidment, D. Tarboton, Y. Liu, P. Passalacqua (2018), GeoFlood: Large scale flood inundation mapping based on high resolution terrain analysis, Water Resources Research, 54, 12, 10013-10033, doi:10.1029/2018WR023457.

When using GeoNet, please cite the following papers:

Passalacqua, P., T. Do Trung, E. Foufoula-Georgiou, G. Sapiro, W. E. Dietrich (2010), A geometric framework for channel network extraction from lidar: Nonlinear diffusion and geodesic paths, Journal of Geophysical Research Earth Surface, 115, F01002, doi:10.1029/2009JF001254.

Sangireddy, H., R. A. Carothers, C.P. Stark, P. Passalacqua (2016), Controls of climate, topography, vegetation, and lithology on drainage density extracted from high resolution topography data, Journal of Hydrology, 537, 271-282, doi:10.1016/j.jhydrol.2016.02.051.
