import pytest, torch
pytest.importorskip("triton")
from examples.vector_add import add
from examples.fused_softmax import fused_softmax

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_vector_add_odd_size():
    x=torch.randn(10007,device="cuda"); y=torch.randn_like(x)
    torch.testing.assert_close(add(x,y),x+y)

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_softmax_odd_columns():
    x=torch.randn(17,513,device="cuda")
    torch.testing.assert_close(fused_softmax(x),torch.softmax(x,-1),rtol=1e-4,atol=1e-5)
