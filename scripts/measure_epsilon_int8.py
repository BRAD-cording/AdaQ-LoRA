"""measure_epsilon_int8.py — INT8-RTN epsilon measurement, runs on single GPU ~2 min."""
import torch, json, gc, os, sys
from transformers import AutoModelForCausalLM
from collections import OrderedDict
import numpy as np

MODEL = os.environ.get("ADAQ_MODEL_PATH", "models/Meta-Llama-3-8B")
print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"":0})
model.eval()

def int8_rtn(w):
    w32 = w.float(); mx = w32.abs().max()
    if mx == 0: return w
    scale = mx / 127.0
    return (torch.clamp(torch.round(w32/scale), -127, 127).float() * scale).bfloat16()

layers = OrderedDict()
for name, param in model.named_parameters():
    if param.dim() < 2: continue
    w_orig = param.detach().float(); norm = w_orig.norm().item()
    if norm == 0: continue
    w_int8 = int8_rtn(param.detach())
    eps = (w_int8.float() - w_orig).norm().item() / norm
    layers[name] = {"eps_int8": round(eps, 8), "norm": round(norm, 4)}

del model; gc.collect(); torch.cuda.empty_cache()

eps_vals = [v["eps_int8"] for v in layers.values()]
a = np.array(eps_vals)
report = {"n": len(eps_vals), "mean": float(a.mean()), "std": float(a.std()),
    "cv_pct": float(a.std()/a.mean()*100), "min": float(a.min()), "max": float(a.max()),
    "ratio": float(a.max()/a.min()), "layers": layers}
with open("results_v2/epsilon_report_int8.json","w") as f: json.dump(report, f, indent=2)
print(f"INT8-RTN: n={len(eps_vals)} mean={a.mean():.6f} std={a.std():.6f} CV={a.std()/a.mean()*100:.2f}% min={a.min():.6f} max={a.max():.6f} ratio={a.max()/a.min():.3f}")

# Compare with existing NF4 and INT4-RTN
for ref_name, ref_path in [("NF4", "results_v2/epsilon_report_nf4.json"),
                             ("INT4-RTN", "results_v2/epsilon_report_int4rtn_g128.json")]:
    if os.path.exists(ref_path):
        with open(ref_path) as f: ref = json.load(f)
        cv_ref = ref.get("cv", ref.get("cv_pct", ref.get("stats",{}).get("cv",0)))
        print(f"  {ref_name} CV (ref): {cv_ref:.2f}%")
print("Saved to results_v2/epsilon_report_int8.json")