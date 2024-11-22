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

**pygeoflood** currently supports Python 3.10, 3.11, and 3.12. We recommend installing with our provided conda environment file. This will install all dependencies and a compatible version of python.

.. code-block:: bash

   conda env create -f pygeoflood-env.yml

Alternatively, you can install the package with pip:

.. code-block:: bash

   pip install git+https://github.com/passah2o/pygeoflood

Examples
#################

.. toctree::
   :maxdepth: 2

.. nbgallery::
   examples/fim_workflow_combined
   examples/fim_workflow_indiv_steps

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


.. Acknowledgements
.. ################



* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
