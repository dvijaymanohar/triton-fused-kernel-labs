# IO-aware fusion

Fusion can reduce intermediate reads/writes and launch overhead, but it can also raise register pressure or reduce scheduling flexibility.

For every fused kernel:
1. identify intermediates removed
2. quantify memory traffic conceptually or with profiler counters
3. verify numerical correctness
4. compare end-to-end and kernel-only timing
5. inspect register use/occupancy where relevant
6. document shapes where fusion stops helping

FlashAttention-style reasoning is about reorganizing attention to reduce expensive memory traffic while preserving exact/controlled semantics; do not reduce it to "one big fused kernel."
