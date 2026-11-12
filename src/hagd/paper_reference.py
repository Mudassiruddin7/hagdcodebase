"""Numbers reported in the paper (arXiv:2601.12879v2), transcribed from its tables.

They are used ONLY to print a "paper" column next to the values this code measures.
Nothing here feeds into any computation.
"""

# Table II - modular arithmetic: (model, method) -> (nodes, edges, preservation, seconds)
TABLE2 = {
    ("gpt2-small", "exhaustive"): (47, 312, 0.98, 14400),
    ("gpt2-small", "acdc"): (52, 487, 0.95, 2340),
    ("gpt2-small", "hierarchical"): (49, 341, 0.97, 180),
    ("gpt2-medium", "acdc"): (78, 892, 0.91, 8720),
    ("gpt2-medium", "hierarchical"): (71, 623, 0.94, 420),
    ("pythia-1.4b", "acdc"): (None, None, None, "OOM"),
    ("pythia-1.4b", "hierarchical"): (124, 1847, 0.91, 1260),
    ("pythia-2.8b", "hierarchical"): (156, 2534, 0.89, 2840),
    ("llama-7b", "hierarchical"): (198, 3412, 0.87, 4520),
    ("llama-70b", "hierarchical"): (347, 8923, 0.82, 28400),
}

# Table III - natural language tasks: (model, task) -> (nodes, edges, preservation, modularity)
TABLE3 = {
    ("gpt2-large", "winogrande"): (312, 4521, 0.86, 0.72),
    ("pythia-2.8b", "winogrande"): (423, 7834, 0.83, 0.68),
    ("llama-7b", "winogrande"): (487, 9123, 0.81, 0.65),
    ("gpt2-large", "hellaswag"): (456, 8912, 0.79, 0.61),
    ("pythia-2.8b", "hellaswag"): (612, 14234, 0.76, 0.58),
    ("llama-7b", "hellaswag"): (734, 18456, 0.74, 0.55),
    ("pythia-2.8b", "factual_qa"): (234, 3412, 0.88, 0.78),
    ("llama-7b", "factual_qa"): (312, 4891, 0.85, 0.74),
    ("llama-70b", "factual_qa"): (489, 9234, 0.81, 0.69),
}

# Table IV - transfer coefficients: (source family, target family) -> {task: tau}
TABLE4 = {
    ("gpt2", "pythia"): {"mod_arith": 0.71, "factual_qa": 0.54, "winogrande": 0.43},
    ("gpt2", "llama"): {"mod_arith": 0.68, "factual_qa": 0.49, "winogrande": 0.38},
    ("pythia", "llama"): {"mod_arith": 0.73, "factual_qa": 0.61, "winogrande": 0.52},
    ("llama-7b", "llama-70b"): {"mod_arith": 0.82, "factual_qa": 0.74, "winogrande": 0.67},
}

# Table V - ablations: configuration -> (mod-arith preservation, QA preservation, relative time)
TABLE5 = {
    "full": (0.91, 0.85, 1.0),
    "no_hierarchy": (0.89, 0.81, 0.3),
    "no_gnn": (0.88, 0.82, 0.7),
    "no_cross_layer": (0.84, 0.76, 1.1),
    "no_causal_validation": (0.91, 0.85, 1.4),
}

# Table VI - scalability: scale -> (memory GB, time hours, features/s, edges/s)
TABLE6 = {
    "100M": (4, 0.05, 12400, 45200),
    "1B": (24, 0.35, 8900, 31400),
    "10B": (96, 4.2, 4200, 12800),
    "70B": (320, 28.4, 1800, 4100),
}

HEADLINE = {
    "reconstruction_unexplained_variance": "15-20%",
    "preservation_algorithmic": "82%-97%",
    "preservation_language": "74%-88%",
    "circuit_nodes": "49-734",
    "transfer": "0.38-0.82",
    "attribution_patching_gap": 0.08,  # preservation points below hierarchical extraction (Section VI-B)
}


def scale_bucket(params: float) -> str:
    """Nearest Table VI scale label for a parameter count."""
    for label, limit in (("100M", 5e8), ("1B", 5e9), ("10B", 4e10)):
        if params < limit:
            return label
    return "70B"
