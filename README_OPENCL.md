# Jlama GPU Inference Engine (OpenCL)

Full-graph GPU inference for Llama-family models using OpenCL. All computation stays on GPU — only token IDs are transferred between CPU and GPU.

## Performance

| Model | GPU (RTX 4090) | CPU (Panama Vector) | Speedup |
|:------|:---------------|:--------------------|:--------|
| Llama 3.2 1B Q4 | **14 tok/s** | 11 tok/s | **+27%** |
| Llama 3.2 3B Q4 | **6 tok/s** | 4 tok/s | **+50%** |

Weight upload time: ~500ms (1B), ~1.2s (3B), ~10s (8B)

## Architecture

```
CPU                              GPU (persistent buffers)
─────                            ────────────────────────
token_id ──→ embed lookup ──→ [hidden_state F32]
                                 │
                                 ├─ RMSNorm kernel
                                 ├─ Q/K/V projection (Q4 matvec kernel)
                                 ├─ RoPE kernel (batched, all heads)
                                 ├─ KV cache store
                                 ├─ Attention kernel (per-head: scores + softmax + weighted sum)
                                 ├─ O projection (Q4 matvec kernel)
                                 ├─ Residual add kernel
                                 ├─ RMSNorm kernel
                                 ├─ Gate + Up projection (Q4 matvec kernels)
                                 ├─ SiLU activation kernel
                                 ├─ Down projection (Q4 matvec kernel)
                                 ├─ Residual add kernel
                                 │  (repeat for all layers)
                                 ├─ Final RMSNorm
                                 ├─ LM Head (Q4 matvec, vocab_size rows)
                                 │
logits[] ←──────────────────────── [logits F32]
    │
    └─ top-k sampling (CPU)
    │
next_token_id
```

### Data Transfer Per Token
- **CPU → GPU**: 8KB (one embedding row, Q4 dequantized on CPU)
- **GPU → CPU**: 500KB (logits for top-k sampling)
- **Between layers**: 0 bytes (everything stays in GPU buffers)

## File Structure

```
jlama-core/src/main/java/com/github/tjake/jlama/tensor/operations/opencl/
├── GpuInferenceEngine.java    # Full transformer forward pass orchestrator
├── GpuModelRunner.java        # Standalone runner (loads model, uploads weights, generates)
└── JOCLTensorOperations.java  # JOCL initialization + GPU detection

jlama-core/src/main/resources/opencl/
├── ops.cl                     # All GPU kernels (main file)
├── gemm.cl                    # F32 matvec/GEMM kernels
└── gemm_q4.cl                 # Q4 matvec kernel (standalone, for reference)
```

## OpenCL Kernels (`ops.cl`)

| Kernel | Purpose | Work Size |
|:-------|:--------|:----------|
| `rmsnorm` | RMS normalization with learnable scale | 256 (single work-group, reduction) |
| `rope` | Rotary Position Embedding (HF halved format) | numHeads × headDim/2 |
| `silu_mul` | SiLU(gate) × up activation | intermediate_size |
| `softmax` | In-place softmax with parallel reduction | 256 (single work-group) |
| `residual_add` | Skip connection accumulation | hidden_size |
| `embedding_lookup` | Copy one row from F32 embedding table | hidden_size |
| `argmax` | Find max index in logits | 1 (single work-item) |
| `matvec_q4` | Q4 dequantize + matrix-vector multiply | output_rows |
| `matvec_f32` | F32 matrix-vector multiply | output_rows |
| `attention_head` | Full attention for one head (scores + softmax + V sum) | 128 (single work-group) |

## Q4 Format on GPU

Weights are stored in Jlama's Q4 format directly on GPU:

```
Q4 Block (32 weights):
┌─────────────────────────────────────────┐
│ 16 bytes (packed nibbles)               │  → 32 weights, 2 per byte
│ 1 float (scale)                         │  → shared by all 32 weights
└─────────────────────────────────────────┘

Dequantization (inside kernel):
  low_nibble  = byte & 0x0F
  high_nibble = (byte >> 4) & 0x0F
  weight_lo   = (low_nibble - 8) * scale    → positions [0..15]
  weight_hi   = (high_nibble - 8) * scale   → positions [16..31]
```

### GPU Memory Layout

| Buffer | Size (1B model) | Size (3B model) |
|:-------|:----------------|:----------------|
| Q4 weight bytes (all layers) | ~500 MB | ~1.5 GB |
| Q4 scales (all layers) | ~60 MB | ~180 MB |
| Embedding (Q4 bytes + scales) | ~125 MB | ~190 MB |
| KV cache (F32, 2048 seq len) | ~32 MB | ~75 MB |
| Activation buffers (F32) | ~1 MB | ~1 MB |
| **Total GPU memory** | **~720 MB** | **~1.95 GB** |

## RoPE Implementation

Llama uses the "halved" RoPE format (HuggingFace convention):

```
For head_dim = 64:
  x0 = vec[offset + 0..31]      (first half)
  x1 = vec[offset + 32..63]     (second half)

  Rotation:
    out[i]      = x0[i] * cos(θ) - x1[i] * sin(θ)
    out[i + 32] = x0[i] * sin(θ) + x1[i] * cos(θ)

  where θ = position / (theta_base^(2i/dim))
  theta_base = 500000.0 for Llama 3.x
```

**Not** the interleaved format `(vec[0], vec[1]), (vec[2], vec[3])...`

## Attention (GQA)

Grouped Query Attention with full GPU computation per head:

```
Llama 3.2 1B: 32 Q heads, 8 KV heads → 4 Q heads share 1 KV head
Llama 3.2 3B: 24 Q heads, 8 KV heads → 3 Q heads share 1 KV head

For each Q head h:
  kv_head = h / (num_heads / num_kv_heads)
  
  1. scores[pos] = dot(Q[h], K_cache[pos][kv_head]) / sqrt(head_dim)
  2. softmax(scores)
  3. output[h] = sum(scores[pos] * V_cache[pos][kv_head])
```

All done in a single kernel launch per head (128 threads, local memory for scores).

## Running

### Quick Test
```bash
cd jlama_gemma4
run_gpu_engine.bat
```

### Benchmark (CPU vs GPU)
```bash
cd jlama_gemma4
benchmark.bat
```

### Programmatic
```java
GpuInferenceEngine engine = new GpuInferenceEngine(config);
engine.initialize();
// Upload weights...
int nextToken = engine.forward(tokenId, position);
```

### From Command Line
```bash
java --add-modules jdk.incubator.vector --enable-preview \
  -cp jlama-cli.jar \
  com.github.tjake.jlama.tensor.operations.opencl.GpuModelRunner \
  /path/to/model "Your prompt here" 100
```

## Supported Models

| Model | Status | Notes |
|:------|:-------|:------|
| Llama 3.2 1B Q4 | ✅ Working | 14 tok/s, correct output |
| Llama 3.2 3B Q4 | ✅ Working | 6 tok/s, correct output |
| Llama 3.1 8B Q4 | ⚠️ WIP | Loads but output quality issues (head_dim=128) |
| Qwen 2.5 Q4 | 🔲 Untested | Should work (same architecture) |
| Mistral 7B Q4 | 🔲 Untested | Should work (same architecture) |
| Gemma 3/4 | ❌ Not supported | Different architecture (multimodal, QK norm) |

## Known Limitations

1. **8B models**: Output quality degrades — likely a head_dim=128 issue in attention or RoPE
2. **Floating point precision**: GPU and CPU produce slightly different results due to accumulation order. Specific prompts give correct answers; vague prompts may diverge
3. **No chat template**: Raw completion mode only (no `<|begin_of_text|>` wrapping)
4. **Embedding lookup**: Done on CPU (Q4 dequant of one row per token, 8KB)
5. **Context length**: Fixed at 2048 tokens (attention kernel local memory limit)
6. **Sampling**: Top-k (k=40, temp=0.6) — no repetition penalty yet

## Dependencies

- **Java 23** (with `jdk.incubator.vector` and `--enable-preview`)
- **JOCL 2.0.6** (Java OpenCL bindings)
- **OpenCL 1.2+** GPU driver (NVIDIA, AMD, or Intel)
- **GPU**: Any OpenCL-compatible GPU with 2GB+ VRAM

## How It Differs from Per-Call GPU

Previous approach (failed):
```
Per token: 10 chunks × 16 layers × (alloc + copy_in + kernel + copy_out + free) = 3000+ transfers
Result: Slower than CPU due to PCIe transfer overhead
```

Current approach (working):
```
Per token: 1 embed write + ~200 kernel launches + 1 logits read = 2 transfers
Result: Faster than CPU, GPU stays busy with compute
```

The key insight: **never move intermediate results off the GPU**. Weights uploaded once, activations persist in GPU buffers, KV cache grows on GPU. Only the final logits come back for sampling.
