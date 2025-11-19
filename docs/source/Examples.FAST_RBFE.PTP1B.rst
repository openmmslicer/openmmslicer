Protein Tyrosine Phosphatase 1B (PTP1B)
===============

Here we perform an alchemical perturbation between two PTP1B ligands (-COMe to -SO2Me), both when solvated and when bound to the protein. Additionally, the optional '-e' flag can enable the enhanced sampling of the four closest torsions to the perturbed groups.

The following script depends on a coordinate and a topology file for each ligand.

**Bound systems**
  - LigA :download:`coordinate <../../examples/PTP1B/bound/LigA.gro>` , :download:`topology <../../examples/PTP1B/bound/LigA.top>`
  - LigB :download:`coordinate <../../examples/PTP1B/bound/LigB.gro>` , :download:`topology <../../examples/PTP1B/bound/LigB.top>`

**Solvated systems**
  - LigA :download:`coordinate <../../examples/PTP1B/free/LigA.gro>` , :download:`topology <../../examples/PTP1B/free/LigA.top>`
  - LigB :download:`coordinate <../../examples/PTP1B/free/LigB.gro>` , :download:`topology <../../examples/PTP1B/free/LigB.top>`

.. literalinclude:: ../../examples/PTP1B/script.py
    :language: python
