# BrainMoE: Spatial Mixture-of-Experts

An experimental small language model inspired by the topographic layout of the brain. Each token is embedded as a small 2D "image" instead of a flat vector. A CNN router picks experts that sit on a 2D grid, and a topographic loss pushes neighbouring experts to behave alike. The goal is to see whether related tokens end up routed to nearby regions of the expert map.

Everything lives in a single notebook: [`Main.ipynb`](Main.ipynb).

## How it works

### Spatial embeddings
Each token maps to a `C × E × E` grid (default `2 × 16 × 16`, so `d_model = 512`), not a flat vector. The output layer reuses the embedding matrix (tied weights).

### Model layout
```
tokens → SpatialEmbedding + learned positions
       → DenseBlock × n_dense          (standard pre-LN transformer block)
       → SpatialBlock × n_spatial      (attention + spatial MoE)
       → LayerNorm → tied output projection
```

### SpatialBlock
1. **Causal self-attention** on the flattened grid, with a residual connection.
2. The grid is reshaped back to `C × E × E` and normalised with `GroupNorm`.
3. **SpatialRouter**: the grid is average-pooled to the expert map size (`8 × 8`). A small CNN with circular padding then produces one logit per expert. The top-k experts (default 4) are kept and their weights renormalised with a softmax.
4. **ConvExperts**: 64 experts arranged on an `8 × 8` torus, implemented as one grouped convolution. Each expert is a 3×3 conv up-projection, GELU, then a 3×3 conv down-projection. The output is the routing-weighted sum of the experts' outputs, added back as a residual.

> Note: right now every expert is computed for every token and the non-selected ones are zeroed by the routing weights. This is fine for studying routing behaviour, but it does not yet give the compute savings of a sparse MoE.

### Losses
- **Cross-entropy** for next-token prediction.
- **Load balancing** (Switch-Transformer style): `K · Σ dispatch · importance − 1`, weighted by `balance_coeff = 0.01`.
- **Topographic kernel loss**: the cosine similarity between two experts' (mean-centred) outputs should match a Gaussian of their distance on the torus, `exp(−d² / 2σ²)`. Weighted by `topo_coeff = 1.0`.
- **Topo ratio** (monitoring only): mean neighbour distance divided by mean all-pairs distance. `1.0` means no spatial structure; lower means neighbouring experts are more alike.

## Training setup

| Setting | Value |
|---|---|
| Dataset | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), first 1% of train; full validation split |
| Tokenizer | GPT-2 (EOS appended to each story) |
| Model | 1 dense + 2 spatial blocks, 8 heads, 64 experts, top-4, **31.1M params** |
| Batch / context | 16 × 128 tokens |
| Optimiser | AdamW (fused), weight decay 0.1 on matrices only, grad clip 1.0 |
| LR schedule | Warmup-stable-decay: 200 warmup steps, peak `1e-3`, linear decay over the last 15% |
| Precision | bf16 autocast on CUDA, `torch.compile` |
| Length | 1 epoch (~2.3k steps) |

Checkpoints are saved to `checkpoints_v4/` every `ckpt_every` steps.

## Analysis cells

After training, the notebook includes:
- **Loss curves** for train and validation cross-entropy.
- **Routing heatmaps**: the router's probability map over the 8×8 expert grid for words such as `the`, `cat`, `dog`, `happy` and `sad`, with the chosen experts marked.
- **Text generation** with temperature and top-k sampling.
- **Expert usage stats**: how many experts are used and how evenly the load is spread.
- **Word-pair routing overlap**: how many experts two words share, and their average torus distance.
- **Expert diversity and topography**: effective dimensionality of the differences between experts, and similarity by grid distance compared with the target kernel.

### Example results (short 1-epoch run)
- Validation CE fell from 10.87 to about 3.87 within the first 250 steps.
- All 64/64 experts were used in both spatial layers. The busiest expert got 3.8–5.0% of picks (an even split would be 1.6%).
- Related words share more experts and route to closer regions:

| Pair | Shared experts (of 4) | Avg. distance to nearest |
|---|---|---|
| `cat` vs `dog` | 2 | 0.75 – 1.25 |
| `happy` vs `sad` | 1–2 | 1.50 |
| `cat` vs `the` | 0 | 2.25 – 3.00 |
| `dog` vs `then` | 0 | 1.25 – 1.50 |

These are early results from a very small run, not a rigorous evaluation.

## Getting started

```bash
pip install torch transformers datasets matplotlib jupyter
jupyter notebook Main.ipynb
```

Run the cells from top to bottom. A CUDA GPU is strongly recommended, since the training loop uses bf16 autocast and a fused AdamW. The dataset and tokenizer download automatically from the Hugging Face Hub.

Main hyperparameters (set in the config cell and `SpatialMoE.__init__`):

| Name | Default | Meaning |
|---|---|---|
| `n_dense` / `n_spatial` | 1 / 2 | Number of dense and spatial blocks |
| `embed_size`, `embed_channels` | 16, 2 | Token grid shape `C × E × E` |
| `expert_size` | 8 | Expert map is `expert_size²` experts on a torus |
| `top_k` | 4 | Experts selected per token |
| `expert_hidden` | 8 | Hidden channels per expert |
| `balance_coeff`, `topo_coeff` | 0.01, 1.0 | Auxiliary loss weights |
