OpenMMSLICER
============

About
-----

OpenMMSLICER (Sequential LIgand Conformational ExploreR) is a Python library which uses OpenMM to perform Fully Adaptive Simulated Tempering (FAST) simulations to enhance the sampling of specific degrees of freedom using an alchemical approach. 
FAST combines adaptive alchemical sequential Monte Carlo (AASMC) with a variation of the irreversible simulated tempering algorithm (IST) that continuously optimizes the number, parameters, and weights of intermediate distributions.

Installation
------------

This package requires OpenMM and OpenMMTools. Afterwards, it can be installed by running ``setup.py``:

.. code-block:: bash

    python setup.py install


Getting Started
---------------

Full docstring documentation can be found `here <https://openmmslicer.readthedocs.io/en/latest/index.html>`_.
There are also a few `examples <https://openmmslicer.readthedocs.io/en/latest/Examples.html>`_ which you can run to
see how OpenMMSLICER works.

Contact
---------------
If you have any problems or questions, please send an email to Justina Ratkeviciute at ``<jr1u18@soton.ac.uk>``.

References
---------------
1. \M. Suruzhon, M. S. Bodnarchuk, A. Ciancetta, I. D. Wall, and J. W. Essex, “Enhancing ligand and protein sampling using sequential Monte Carlo,” *J. Chem. Theory Comput.* 18, 3894–3910 (2022), DOI:https://doi.org/10.1021/acs.jctc.1c01198
2. \M. Suruzhon, K. Abdel-Maksoud, M. S. Bodnarchuk, A. Ciancetta, I. D. Wall, and J. W. Essex, "Enhancing torsional sampling using fully adaptive simulated tempering," *J. Chem. Phys.* 160, 154110 (2024), DOI:https://doi.org/10.1063/5.0190659
