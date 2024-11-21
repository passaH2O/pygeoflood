Contributing
============

Bug reports
-----------

We welcome contributions to **pygeoflood**. Please file bug reports on the `GitHub issue tracker <https://github.com/passaH2O/pygeoflood/>`_.

Development
-----------

To obtain an editable version of the code, clone and install in editable mode:

.. code-block:: bash

    git clone https://github.com/passah2o/pygeoflood
    cd pygeoflood
    pip install -e .

To submit a pull request, fork the repository and create a new branch with your changes. Then, submit a pull request to the main branch.

Building the docs
-----------------

To build the documentation, use the provided environment:

.. code-block:: bash

    conda env create -f doc/pgfdocs-env.yml
    conda activate pgfdocs-env
    cd doc
    make html