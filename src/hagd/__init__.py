"""Hierarchical Attribution Graph Decomposition (HAGD).

Replication of "Hierarchical Sparse Circuit Extraction from Billion-Parameter Language
Models through Scalable Attribution Graph Decomposition" (arXiv:2601.12879).

The four stages of the method map onto modules as follows:

    cross-layer transcoder training      -> hagd.transcoder, hagd.train_transcoder
    attribution graph construction       -> hagd.attribution
    spectral hierarchical decomposition  -> hagd.hierarchy
    GNN-guided hierarchical traversal    -> hagd.gnn, hagd.search
    causal intervention verification     -> hagd.validation
"""

__version__ = "0.1.0"
