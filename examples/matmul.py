import torch, triton, triton.language as tl

@triton.jit
def mm_kernel(a,b,c,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    rm=pid_m*BM+tl.arange(0,BM); rn=pid_n*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,K,BK):
        ak=k0+rk
        av=tl.load(a+rm[:,None]*K+ak[None,:],mask=(rm[:,None]<M)&(ak[None,:]<K),other=0.0)
        bv=tl.load(b+ak[:,None]*N+rn[None,:],mask=(ak[:,None]<K)&(rn[None,:]<N),other=0.0)
        acc+=tl.dot(av,bv)
    tl.store(c+rm[:,None]*N+rn[None,:],acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

def matmul(a,b):
    assert a.shape[1]==b.shape[0] and a.is_cuda and b.is_cuda
    M,K=a.shape; _,N=b.shape; c=torch.empty((M,N),device=a.device,dtype=a.dtype)
    grid=(triton.cdiv(M,32),triton.cdiv(N,32))
    mm_kernel[grid](a,b,c,M,N,K,BM=32,BN=32,BK=32,num_warps=4)
    return c

if __name__=="__main__":
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    torch.manual_seed(0)
    a=torch.randn(127,129,device="cuda",dtype=torch.float16)
    b=torch.randn(129,131,device="cuda",dtype=torch.float16)
    got=matmul(a,b); ref=a@b
    torch.testing.assert_close(got,ref,rtol=2e-2,atol=2e-2)
    print("PASS",got.shape)
