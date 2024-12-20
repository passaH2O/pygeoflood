.. pygeoflood documentation master file, created by
   sphinx-quickstart on Fri May 24 10:07:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**pygeoflood** docs
========================

**pygeoflood** is a suite of terrain-analysis tools for mapping flood inundation in near-real time.

This package is under active development as we incorporate efficient methods for mapping fluvial, pluvial, and coastal flooding.

View the source code on GitHub: https://github.com/passah2o/pygeoflood

.. implementation of geonet and geoflood, which are topographic based channel extraction packages that delineate slope, curvature, flow direction, and flow accumulation in order to implement the height above nearest drainage (HAND) inundation mapping method.

.. This package is under active development as we build out additional methods to map compound flooding in near-real time.

Installation
#############

We recommend installing **pygeoflood** into a conda environment. Python 3.10, 3.11, and 3.12 are currently supported:

.. code-block:: bash

   pip install pygeoflood

For the bleeding-edge version, install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/passah2o/pygeoflood

Example workflows in python
###########################

.. toctree::
   :maxdepth: 2

   Example gallery <ex>

.. Getting Started
.. ###############

.. Background
.. ##########

.. User Guide
.. ##########

Contributing
#############

.. toctree::
   :maxdepth: 1

   Contributing bug reports and development <contribute>

API Reference
#############

.. toctree::
   :maxdepth: 1

   pygeoflood API reference <apiref/apiref>


Acknowledgements
################

.. toctree::
   :maxdepth: 1

   Acknowledgements <acknowledgements>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
