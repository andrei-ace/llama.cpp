# TurboQuant — Exact Paper Algorithms (from turboquant.net + arXiv 2504.19874)

## PolarQuant: polar-coordinate transform

The key idea: remove per-block normalization overhead. PolarQuant rotates the vector
randomly so coordinates follow a concentrated distribution that is easy to quantize.

**Coordinate distribution** (after rotation, for unit-norm input):
```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) × (1 - x²)^((d-3)/2)
where x ∈ [-1, 1]
```
This is `Beta((d-1)/2, (d-1)/2)` on [-1,1]. In high dimensions, it approximates `N(0, 1/d)`.

**Steps:**
1. Group the d-dimensional vector into pairs to obtain radii and angles
2. Apply recursive polar transforms on the radii
3. Quantize only the angles, whose distribution is highly concentrated

**Why it works:**
- No per-block full-precision constants — overhead drops to zero
- Near-lossless beyond 4.2× compression
- Gaussian-like coordinates in high dimension → supports Lloyd-Max

**Critical detail:** PolarQuant is NOT just a random rotation. It separates the vector
into radius (‖x‖) + angles via a polar coordinate transform. The angles follow
the Beta distribution and are quantized. The radius is handled implicitly — this is
how "no per-block constants" is achieved. A plain QR rotation does NOT give this
property; you must store the norm separately.

## Algorithm 1: TurboQuant_mse (optimized for MSE)

```
Input: dimension d, bit width b

// Global setup (once):
1. Generate random rotation matrix Π ∈ R^{d×d}
2. Construct codebook: centroids c_1,...,c_{2^b} ∈ [-1,1]
   that minimize Lloyd-Max MSE for the Beta((d-1)/2,(d-1)/2) distribution

// QUANT_mse(x):
3. y ← Π · x                                    // rotate (NO normalization)
4. idx_j ← argmin_k |y_j - c_k|  for j ∈ [d]   // idx_j are b-bit integers
5. Output: idx

// DEQUANT_mse(idx):
6. ỹ_j ← c_{idx_j}  for j ∈ [d]                // look up centroids
7. x̃ ← Π^T · ỹ                                 // rotate back
8. Output: x̃
```

**Key:** NO normalization step. The centroids are pre-scaled for the Beta distribution
which has variance ~1/d. For b=1: centroids = {±√(2/πd)}. For b=2: centroids =
{±0.453/√d, ±1.51/√d}.

**MSE upper bound:** `D_MSE ≤ (√3 · π/2) · 1/4^b`

**Important:** This algorithm assumes the input lies on (or near) the unit sphere,
OR that PolarQuant handles the radius separately. Without PolarQuant, you MUST
normalize and store the norm — which is what our current implementation does.

## Algorithm 2: TurboQuant_prod (optimized for inner product)

```
Input: dimension d, bit width b

// Global setup (once):
1. Instantiate TurboQuant_mse with (b-1) bits
2. Generate random projection matrix S ∈ R^{d×d} with S_ij ~ N(0,1)

// QUANT_prod(x):
3. idx ← QUANT_mse(x)                           // MSE quantize with b-1 bits
4. x̄_mse ← DEQUANT_mse(idx)                     // full MSE reconstruction
5. r ← x - x̄_mse                                // residual in ORIGINAL space
6. qjl ← sign(S · r)                            // QJL on residual
7. Output: (idx, qjl, ‖r‖₂)

// DEQUANT_prod(idx, qjl, γ):
8. x̃_mse ← DEQUANT_mse(idx)                     // MSE reconstruction
9. x̃_qjl ← √(π/2)/d · γ · S^T · qjl           // QJL correction
10. Output: x̃_mse + x̃_qjl
```

**Inner-product upper bound:** `D_prod ≤ (π²√3 · ‖y‖²/d) · 1/4^b`

## QJL: 1-bit unbiased inner-product estimator

**Quantization:** `Q_qjl(r) = sign(S · r)` where `S_ij ~ N(0,1)`

**Dequantization:** `r̂ = √(π/2d) · S^T · Q_qjl(r)`

Note: the website formula is `r̂ = √(π/2d) · S^T · Q_qjl(r)` with the residual
norm ‖r‖ multiplied in during DEQUANT_prod (the γ parameter).

**Unbiasedness:** `E[⟨y, r̂⟩] = ⟨y, r⟩`

**Variance bound:** `Var ≤ (π/2d) · ‖y‖² · ‖r‖²`

## Implementation sketch (from website)

```python
# Step 1: Precompute Lloyd-Max centroids
centroids = lloyd_max_quantizer(distribution="beta", bits=b)

# Step 2: Generate random rotation matrix
G = np.random.randn(d, d)
Pi, _ = np.linalg.qr(G)

# Step 3: Build quant/dequant primitives
def quant(x, Pi, centroids):
    y = Pi @ x
    idx = find_nearest(y, centroids)
    return idx

def dequant(idx, Pi, centroids):
    y = centroids[idx]
    x = Pi.T @ y
    return x

# Step 4: Integrate inside attention
k_quant = turboquant_quant(k)
v_quant = turboquant_quant(v)
# use QJL during attention
```

## Differences from our current implementation

| Aspect | Paper | Our code |
|--------|-------|----------|
| **Norm handling** | PolarQuant separates radius implicitly — zero overhead | Explicit `norm = ‖x‖`, store as fp16 per block |
| **Rotation** | Π applied to raw x (no normalization) | Π applied to x/‖x‖ (normalized first) |
| **Residual space** | `r = x - DEQUANT_mse(idx)` — original space | `r = x_normalized - centroid` — normalized rotated space |
| **QJL dequant** | `√(π/2d) · γ · S^T · qjl` (γ=‖r‖, d in formula) | `√(π/2)/m · rnorm · S^T · signs` (m=dimension, similar) |
| **Per-block overhead** | MSE: 0 bytes. PROD: ‖r‖ only (2 bytes) | MSE: 4 bytes (2×fp16 norms). PROD: 8 bytes (4×fp16) |
| **Outlier split** | Not in base algorithm | We split into 32 outlier + 96 regular channels |
| **Centroids** | Optimal for Beta((d-1)/2,(d-1)/2) on [-1,1] | Same centroids, but applied to normalized coordinates |

## Key insight

The paper's "no per-block constants" claim relies on PolarQuant (true polar
coordinate transform), NOT just a random orthogonal rotation. Without PolarQuant,
our explicit normalization + stored norms is correct and necessary. The centroids
are designed for unit-sphere inputs; if we don't normalize, the rotated coordinates
don't match the centroid scale and quality collapses.

Our current approach (normalize → rotate → quantize → store norm) is a valid
adaptation that works, but uses 4-8 bytes more per block than the paper claims.
To achieve the paper's zero-overhead, we would need to implement the actual
PolarQuant polar coordinate decomposition.
