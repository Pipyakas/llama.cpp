# Pipyakas Native Quant Findings

Research date: 2026-08-17. Fork commit: `1c3f93f`.

## Native weight paths

Unsloth UD files are mixtures of standard GGML types, not new runtime types.
`UD-Q4_K_XL` contains Q4_K, Q8_0, Q5_K, Q6_K, and F32 tensors. On the
fleet's CUDA and x86 CPU backends, these types use native quantized kernels;
the Q2_0_128 type is the fork-specific exception that required a native CUDA
path.

On the RTX 2060 (`sm_75`), the quantized-weight CUDA paths unpack values to
INT8 in registers and use INT8 MMA or DP4A/IMAD. A cubin census found 158,777
IMMA instructions and 98,448 HMMA instructions. The quantized GEMMs account
for the INT8 work, not the HMMA work.

## HMMA boundary

The remaining FP16 tensor-core work is expected from flash attention and
genuinely-F16 tensors such as an F16 vision projector. llama.cpp has no INT8
attention implementation. Disabling flash attention does not make inference
INT8-only; it falls back to slower FP16/FP32 attention. Quantizing an F16
projector to Q8_0 is the practical smaller lever.

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
