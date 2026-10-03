import statistics, time, torch
from examples.vector_add import add

if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
x=torch.randn(1<<24,device="cuda"); y=torch.randn_like(x)
for _ in range(10): add(x,y)
torch.cuda.synchronize()
samples=[]
for _ in range(30):
    a=torch.cuda.Event(True); b=torch.cuda.Event(True)
    a.record(); add(x,y); b.record(); b.synchronize()
    samples.append(a.elapsed_time(b))
print({"median_ms":statistics.median(samples),"min_ms":min(samples),"max_ms":max(samples)})
