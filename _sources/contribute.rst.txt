Contributing
============

Bug reports
-----------

We welcome contributions to **pygeoflood**. Please file bug reports on the `GitHub issue tracker <https://github.com/passaH2O/pygeoflood/>`_.

Development
-----------

Clone and install in editable mode:

.. code-block:: bash

    git clone https://github.com/passah2o/pygeoflood
    cd pygeoflood
    pip install -e .

To contribute: fork, create a new branch with your changes, and submit a pull request.

Building the docs
-----------------

To build the documentation, use the provided environment:

.. code-block:: bash

    conda env create -f doc/pgfdocs-env.yml
    conda activate pgfdocs-env
    cd doc
    make html