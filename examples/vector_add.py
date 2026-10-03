import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x,y,out,n:tl.constexpr,BLOCK:tl.constexpr):
    offs=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=offs<n
    tl.store(out+offs,tl.load(x+offs,mask=mask)+tl.load(y+offs,mask=mask),mask=mask)

def add(x,y):
    out=torch.empty_like(x); n=x.numel()
    grid=(triton.cdiv(n,256),)
    add_kernel[grid](x,y,out,n,BLOCK=256)
    return out

if __name__=="__main__":
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    torch.manual_seed(0); x=torch.randn(10007,device="cuda"); y=torch.randn_like(x)
    got=add(x,y); torch.testing.assert_close(got,x+y)
    print("PASS",got.numel())
