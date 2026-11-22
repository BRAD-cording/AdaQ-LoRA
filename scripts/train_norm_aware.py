#!/usr/bin/env python3
"""Norm-aware baseline: allocates ranks proportional to ||W_l||_F only (no epsilon).
This is the control experiment for M2: if adaq == norm_aware, it confirms score degeneracy.
"""
import json, os, sys, torch
from pathlib import Path

def compute_norm_aware_ranks(model_path, budget=64, r_min=1, r_max=16):
    """Allocate ranks proportional to Frobenius norm of each linear layer."""
    from safetensors.torch import load_file
    
    # Load model index
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No index file at {index_path}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    # Collect norms per module
    norms = {}
    weight_map = index["weight_map"]
    loaded_files = set()
    
    for key, fname in weight_map.items():
        if ".weight" not in key or any(x in key for x in ["embed", "lm_head", "norm"]):
            continue
        # Extract module name (e.g., model.layers.0.self_attn.q_proj)
        module_name = key.replace(".weight", "")
        
        if fname not in loaded_files:
            sf_path = Path(model_path) / fname
            if sf_path.exists():
                state_dict = load_file(str(sf_path))
                loaded_files.add(fname)
            else:
                continue
        
        if key in state_dict:
            w = state_dict[key].float()
            norms[module_name] = w.norm().item()
    
    if not norms:
        raise ValueError("No linear layers found")
    
    # Allocate ranks proportional to norm
    total_norm = sum(norms.values())
    raw_ranks = {k: (v / total_norm) * budget for k, v in norms.items()}
    
    # Clamp and round
    ranks = {}
    for k, v in raw_ranks.items():
        r = max(r_min, min(r_max, round(v)))
        ranks[k] = r
    
    # Adjust to match budget
    current = sum(ranks.values())
    if current != budget:
        # Sort by fractional part to decide which to adjust
        fracs = [(k, raw_ranks[k] - ranks[k]) for k in ranks]
        fracs.sort(key=lambda x: x[1], reverse=(current < budget))
        diff = budget - current
        for i in range(abs(diff)):
            k = fracs[i % len(fracs)][0]
            ranks[k] += 1 if diff > 0 else -1
            ranks[k] = max(r_min, min(r_max, ranks[k]))
    
    return ranks

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Meta-Llama-3-8B"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    
    ranks = compute_norm_aware_ranks(model_path, budget)
    
    # Save
    out = {"method": "norm_aware", "budget": budget, "ranks": ranks}
    out_path = "configs/norm_aware_ranks.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"Saved {len(ranks)} layer ranks to {out_path}")
    print(f"Budget: {budget}, Actual sum: {sum(ranks.values())}")
    print(f"Sample: q_proj layers = {[ranks.get(k) for k in sorted(ranks) if 'q_proj' in k][:4]}")
