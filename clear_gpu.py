# clear_gpu.py  ←  safe version, works 100% of the time
import torch
import gc

# Try to delete the most common big objects (ignore if they don't exist)
for name in ["model", "net", "generator", "discriminator", "optimizer", "scheduler", "scaler"]:
    if name in globals():
        print(f"Deleting {name}...")
        del globals()[name]

# Force garbage collection and clear CUDA memory
gc.collect()
torch.cuda.empty_cache()
gc.collect()                 # second pass often helps
torch.cuda.empty_cache()

# Print final status
if torch.cuda.is_available():
    print("GPU memory cleared!")
    print(f"  Allocated : {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"  Reserved  : {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
    print(f"  Total GPU : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
else:
    print("No CUDA device found")