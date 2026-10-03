import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x,y,n_cols:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0,BLOCK)
    mask=offs<n_cols
    v=tl.load(x+row*n_cols+offs,mask=mask,other=-float("inf"))
    v=v-tl.max(v,axis=0)
    num=tl.exp(v); den=tl.sum(num,axis=0)
    tl.store(y+row*n_cols+offs,num/den,mask=mask)

def fused_softmax(x):
    assert x.ndim==2
    rows,cols=x.shape; block=triton.next_power_of_2(cols)
    y=torch.empty_like(x); softmax_kernel[(rows,)](x,y,cols,BLOCK=block,num_warps=4); return y

if __name__=="__main__":
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    x=torch.randn(256,513,device="cuda")
    y=fused_softmax(x)
    torch.testing.assert_close(y,torch.softmax(x,-1),rtol=1e-4,atol=1e-5)
    print("PASS",tuple(y.shape))
