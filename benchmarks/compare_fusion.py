import statistics, torch, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.fused_bias_relu import fused_bias_relu

if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
x=torch.randn(1<<24,device="cuda"); b=torch.randn_like(x)

def eager(): return torch.relu(x+b)
def triton_fused(): return fused_bias_relu(x,b)

for fn in [eager,triton_fused]:
    for _ in range(10):fn()
    samples=[]
    for _ in range(40):
        a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); z.record(); z.synchronize(); samples.append(a.elapsed_time(z))
    print(fn.__name__,{"median_ms":statistics.median(samples),"min_ms":min(samples),"max_ms":max(samples)})
