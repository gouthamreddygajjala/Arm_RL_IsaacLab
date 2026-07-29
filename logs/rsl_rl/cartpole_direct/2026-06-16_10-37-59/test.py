import torch

ckpt_path = "model_0.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

output_lines = []

def log(line=""):
    print(line)
    output_lines.append(line)

log(f"Top-level type: {type(ckpt)}")
log("")

if isinstance(ckpt, dict):
    log("Top-level keys:")
    for k, v in ckpt.items():
        log(f"  {k}: {type(v)}")
    log("")

    # Drill into every dict-like sub-entry (policy, value, optimizer, state_preprocessor, etc.)
    for k, v in ckpt.items():
        log(f"--- Contents of '{k}' ---")
        if isinstance(v, dict):
            for kk, vv in v.items():
                if hasattr(vv, "shape"):
                    log(f"  {kk}: shape={tuple(vv.shape)} dtype={vv.dtype}")
                else:
                    log(f"  {kk}: {type(vv)} -> {vv if not hasattr(vv, '__len__') or len(str(vv)) < 200 else 'large object'}")
        elif hasattr(v, "shape"):
            log(f"  tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            log(f"  value: {v}")
        log("")
else:
    log("Checkpoint is not a dict — printing raw repr:")
    log(repr(ckpt))

with open("ckpt_structure.txt", "w") as f:
    f.write("\n".join(output_lines))

print("\n\n>>> Full dump also saved to ckpt_structure.txt")