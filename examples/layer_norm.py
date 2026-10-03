import torch, triton, triton.language as tl

@triton.jit
def layer_norm_kernel(x,w,b,y,n_cols:tl.constexpr,eps:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0,BLOCK); mask=offs<n_cols
    v=tl.load(x+row*n_cols+offs,mask=mask,other=0.0)
    mean=tl.sum(v,axis=0)/n_cols
    diff=tl.where(mask,v-mean,0.0)
    var=tl.sum(diff*diff,axis=0)/n_cols
    norm=(v-mean)*tl.rsqrt(var+eps)
    ww=tl.load(w+offs,mask=mask,other=0.0); bb=tl.load(b+offs,mask=mask,other=0.0)
    tl.store(y+row*n_cols+offs,norm*ww+bb,mask=mask)

def layer_norm(x,w,b,eps=1e-5):
    rows,cols=x.shape; block=triton.next_power_of_2(cols)
    y=torch.empty_like(x)
    layer_norm_kernel[(rows,)](x,w,b,y,cols,eps,BLOCK=block,num_warps=4)
    return y

if __name__=="__main__":
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    x=torch.randn(64,513,device="cuda"); w=torch.randn(513,device="cuda"); b=torch.randn(513,device="cuda")
    got=layer_norm(x,w,b)
    ref=torch.nn.functional.layer_norm(x,(513,),w,b)
    torch.testing.assert_close(got,ref,rtol=2e-3,atol=2e-3)
    print("PASS")
