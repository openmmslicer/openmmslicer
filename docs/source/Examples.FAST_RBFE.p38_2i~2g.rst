p38 MAP Kinase
===============

This test case demonstrates the alchemical perturbation between two p38 MAP kinase ligands (-CH3- to -O-), both in solution and when bound to protein. The optional '-e' flag enables enhanced sampling to explore the torsional rotations of the ligand's phenyl ring.

The following script depends on a coordinate and a topology file for each ligand.

**Bound systems**
  - LigA :download:`coordinate <../../examples/p38/bound/LigA.gro>` , :download:`topology <../../examples/p38/bound/LigA.top>`
  - LigB :download:`coordinate <../../examples/p38/bound/LigB.gro>` , :download:`topology <../../examples/p38/bound/LigB.top>`

**Solvated systems**
  - LigA :download:`coordinate <../../examples/p38/free/LigA.gro>` , :download:`topology <../../examples/p38/free/LigA.top>`
  - LigB :download:`coordinate <../../examples/p38/free/LigB.gro>` , :download:`topology <../../examples/p38/free/LigB.top>`

.. literalinclude:: ../../examples/p38/script.py
    :language: python
