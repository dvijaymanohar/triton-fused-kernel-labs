# Expanded Triton labs

Run the sequence:
```bash
python examples/vector_add.py
python examples/fused_bias_relu.py
python examples/fused_softmax.py
python examples/layer_norm.py
python examples/matmul.py
python benchmarks/compare_fusion.py
```

Then inspect generated kernels and profiler metrics. Exercises:
- add autotune configurations to matmul;
- vary BLOCK sizes and num_warps;
- compare eager PyTorch, torch.compile where available, and Triton;
- implement a numerically stable attention forward pass;
- explain cases where fusion reduces memory traffic but raises register pressure.
