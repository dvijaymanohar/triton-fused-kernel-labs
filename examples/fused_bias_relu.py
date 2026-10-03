import torch, triton, triton.language as tl

@triton.jit
def kernel(x,bias,out,n:tl.constexpr,BLOCK:tl.constexpr):
    offs=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=offs<n
    v=tl.load(x+offs,mask=mask)+tl.load(bias+offs,mask=mask)
    tl.store(out+offs,tl.maximum(v,0.0),mask=mask)

def fused_bias_relu(x,bias):
    assert x.shape==bias.shape and x.is_cuda
    out=torch.empty_like(x); n=x.numel()
    kernel[(triton.cdiv(n,256),)](x,bias,out,n,BLOCK=256)
    return out

if __name__=="__main__":
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    x=torch.randn(10007,device="cuda"); b=torch.randn_like(x)
    got=fused_bias_relu(x,b)
    torch.testing.assert_close(got,torch.relu(x+b))
    print("PASS")
