# Pipyakas Native Quant Findings

Research date: 2026-08-17. Fork commit: `1c3f93f`.

> General research (native kernel coverage, HMMA/int8 analysis, CPU-routed
> decode bottleneck + streaming experiments) lives on L0 in
> `NATIVE-QUANTS.md` and `REQUANT-PIPELINE.md`. This file keeps only
> machine-specific benchmark data.

## Benchmark caveat

The first RTX 5070 Ti comparison attempt used `UD-Q4_K_XL` with CPU expert
overrides and speculative MTP. The model loaded and prefilling worked, but
generation emitted EOS immediately (`tokens_predicted: 1`, empty content), so
that run is invalid as a performance result. Benchmarking must first validate
non-empty deterministic output without MTP, then add MTP only if both models
produce equivalent output.

## RTX 5070 Ti comparison

Measured on R3 (RTX 5070 Ti 16 GB) with commit `1c3f93f`, CUDA server,
flash attention enabled, Q8 KV cache, `-c 8192`, and all routed expert
tensors on CPU. Shared and dense layers remained on the GPU. Both models used
the same 15-token prompt and generated 200 tokens without MTP; Q4 MTP was
excluded because it emitted EOS immediately in this configuration.

| Model | File size | Decode | GPU memory | Host RAM observation |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | 22,853,663,008 B | 60.3 t/s | 2,966 MiB | 33.5 GB free |
| expQ2_0_128-attnQ4K_M | 10,765,277,216 B | 88.5 t/s | 1,959 MiB | 44.1 GB free |

The expQ2_0_128 build is 1.47x faster and 52.9% smaller on disk. In this
sequential run it left about 10.6 GB more host RAM free; RAM figures include
Windows file-cache effects. Both outputs produced all 200 requested tokens.

KLD used the existing Q8_0 reference logits from 145 Wikitext chunks x 2048
tokens (`q8_0_kld.logits`). The comparison is independent of expert placement:

| Model | Mean PPL | Mean KLD | RMS delta-p | Same top-p |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | 5.9018 | 0.012671 | 3.443% | 95.716% |
| expQ2_0_128-attnQ4K_M | 18.0836 | 1.258114 | 34.056% | 57.626% |

UD-Q4_K_XL has substantially better quality, while expQ2_0_128 is smaller
and faster when both are constrained to CPU-routed experts.

## Placement sweep (2026-08-17, decode t/s, no MTP)

R3 Q4_K_XL: all-CPU experts 59.6 -> 26 GPU-expert layers 98.2 (14.9 GB VRAM).
D1 our quant: full-GPU 76.5 (fits 12 GB) vs all-CPU 41.6. D1 Q4: all-CPU 35.8
-> 16 GPU 43.9 -> 20 GPU 47.1 (11.9 GB VRAM). L1 our: all-CPU 26.2 -> 24 GPU
46.7. L1 Q4: all-CPU 24.1 -> 9 GPU 31.9 (7.9 GB VRAM, 11 GPU OOMs).
D1 AVX-512 rebuild: no decode change (not SIMD-bound). D1 -t 16 +8%; L1 -t 16
-8% (single-channel). Full detail in L0 REQUANT-PIPELINE.md / NATIVE-QUANTS.md.
