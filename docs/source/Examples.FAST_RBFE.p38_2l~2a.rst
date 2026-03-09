p38 MAP Kinase
===============

This test case demonstrates the alchemical perturbation between two p38 MAP kinase ligands (-CH3- to -O-), both in solution and when bound to protein. The optional '-e' flag enables enhanced sampling to explore the torsional rotations of the ligand's phenyl ring.

The following script depends on a coordinate and a topology file for each ligand.

**Bound systems**
  - LigA :download:`coordinate <../../examples/p38_2l~2a/bound/LigA.gro>` , :download:`topology <../../examples/p38_2l~2a/bound/LigA.top>`
  - LigB :download:`coordinate <../../examples/p38_2l~2a/bound/LigB.gro>` , :download:`topology <../../examples/p38_2l~2a/bound/LigB.top>`

**Solvated systems**
  - LigA :download:`coordinate <../../examples/p38_2l~2a/free/LigA.gro>` , :download:`topology <../../examples/p38_2l~2a/free/LigA.top>`
  - LigB :download:`coordinate <../../examples/p38_2l~2a/free/LigB.gro>` , :download:`topology <../../examples/p38_2l~2a/free/LigB.top>`

.. literalinclude:: ../../examples/p38_2l~2a/script.py
    :language: python
