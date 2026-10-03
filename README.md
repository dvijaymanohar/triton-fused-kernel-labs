# Triton Fused Kernel Labs

Learn Triton by implementing kernels that expose masking, program IDs, fusion, tiling, autotuning, and IO-aware design.

## Sequence
vector add → masked tails → fused elementwise → softmax → layer norm → matmul → autotuning → naive attention → fused attention.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/vector_add.py
python examples/fused_softmax.py
pytest -q
```

Triton requires a supported GPU/runtime. Every kernel is compared against a trusted PyTorch reference.
