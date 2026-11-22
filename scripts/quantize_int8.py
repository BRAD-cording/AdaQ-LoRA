"""quantize_int8.py — Save INT8-RTN dequantized model for training."""
import torch, os, gc, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ.get("ADAQ_MODEL_PATH", "models/Meta-Llama-3-8B")
DST = "models/Meta-Llama-3-8B-int8rtn"
if os.path.exists(DST):
    print(f"{DST} exists, skipping")
    sys.exit(0)

print(f"Quantizing {SRC} -> {DST}")
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map={"":0})
model.eval()

n_params = 0
for name, param in model.named_parameters():
    if param.dim() < 2: continue
    w32 = param.data.float(); mx = w32.abs().max()
    if mx == 0: continue
    scale = mx / 127.0
    quant = torch.clamp(torch.round(w32/scale), -127, 127)
    param.data.copy_((quant.float() * scale).bfloat16())
    n_params += 1
print(f"Quantized {n_params} weight matrices")

os.makedirs(DST, exist_ok=True)
model.save_pretrained(DST)
tokenizer = AutoTokenizer.from_pretrained(SRC)
tokenizer.save_pretrained(DST)
del model; gc.collect(); torch.cuda.empty_cache()
print(f"Saved to {DST}")