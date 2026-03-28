# TurboQuant Website (turboquant.net) — Full Documentation

## Tab 1: PolarQuant

**"PolarQuant: polar-coordinate transform"**

The key idea is to **remove per-block normalization overhead**. PolarQuant rotates the vector randomly so coordinates follow a concentrated distribution that is easy to quantize.

**Coordinate distribution:**
```
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
where x in [-1, 1]
```

**Steps:**
1. Group the d-dimensional vector into pairs to obtain radii and angles
2. Apply recursive polar transforms on the radii
3. Quantize only the angles, whose distribution is highly concentrated

**Why it works:**
- **No per-block full-precision constants** — Overhead drops to zero
- **Near-lossless beyond 4.2x compression** — Stronger than conventional baselines
- **Gaussian-like coordinates in high dimension** — Supports optimal scalar quantizers such as Lloyd-Max

## Tab 2: QJL

**"QJL: a 1-bit unbiased inner-product estimator"**

Johnson-Lindenstrauss projection reduces dimensionality, and QJL stores only the sign of each projected component.

**Quantization formula:**
```
Q_qjl(r) = sign(S * r)
where S_ij ~ N(0,1)
```

**Dequantization formula:**
```
r_hat = sqrt(pi/2d) * S^T * Q_qjl(r)
```

**Unbiasedness guarantee:**
```
E[<y, r_hat>] = <y, r>
```

**Variance bound:**
```
Var <= (pi/2d) * ||y||^2 * ||r||^2
```

**Why it matters:**
- Stores residual information with only 1 bit
- Keeps inner-product estimation unbiased
- Complements PolarQuant cleanly in the second stage

## Tab 3: Full Algorithm

### TurboQuant_mse (MSE-optimized, b bits)

```
Algorithm: TurboQuant_mse
Input: dimension d, bit width b

1. Precompute centroids c_k (Lloyd-Max)
2. For each vector x:
   2.1 y <- Pi * x
   2.2 For j = 1 to d:
       idx_j <- argmin_k |y_j - c_k|
   2.3 Output idx
```

### TurboQuant_prod (Inner-product optimized, b bits)

```
Algorithm: TurboQuant_prod
Input: dimension d, bit width b

1. Instantiate TurboQuant_mse with (b - 1) bits
2. Generate random projection matrix S
3. For each vector x:
   3.1 x_bar_mse <- Q_mse(x)
   3.2 r <- x - x_bar_mse
   3.3 qjl <- sign(S * r)
   3.4 Output (idx, qjl, ||r||_2)
```

### Theoretical guarantees

- **MSE upper bound:** `D_MSE <= (sqrt(3) * pi/2) * 1/4^b`
- **Inner-product upper bound:** `D_prod <= (pi^2 * sqrt(3) * ||y||^2 / d) * 1/4^b`

## Implementation Sketch

### Step 1: Precompute Lloyd-Max centroids
Do it once offline and reuse them.
```python
centroids = lloyd_max_quantizer(distribution="beta", bits=b)
```

### Step 2: Generate a random rotation matrix
Use QR decomposition to build an orthogonal matrix.
```python
G = np.random.randn(d, d)
Pi, _ = np.linalg.qr(G)
```

### Step 3: Build quant / dequant primitives
This is the core path for storage and recovery.
```python
def quant(x, Pi, centroids):
    y = Pi @ x
    idx = find_nearest(y, centroids)
    return idx

def dequant(idx, Pi, centroids):
    y = centroids[idx]
    x = Pi.T @ y
    return x
```

### Step 4: Integrate inside attention
Store K/V in TurboQuant form and estimate inner products with QJL.
```python
k_quant = turboquant_quant(k)
v_quant = turboquant_quant(v)
# use QJL during attention
```

## Deployment Notes

- **Hardware:** H100 and A100 are ideal. 4-bit mode is where the paper reports 8x speedups.
- **Mixed precision:** Use TurboQuant for KV cache and INT4 for weights to maximize total compression.
- **Edge devices:** 3-bit KV cache can make 32K+ context feasible on phones with software-only implementations.

### Practical risks and mitigations
- **Random rotation overhead:** Pre-generate and reuse the matrices instead of rebuilding them online.
- **Residual norm storage:** One FP16 scalar is small enough to keep the overhead negligible.

---

## KEY OBSERVATIONS — Differences from our implementation

### 1. PolarQuant vs QR rotation
The website describes PolarQuant as a **polar coordinate transform** (pairs -> radii + angles -> recursive), NOT just a rotation. This naturally separates direction from magnitude and puts coordinates into [-1, 1] range. Our implementation uses a standard QR rotation which requires explicit normalization.

### 2. No normalization in pseudocode
The TurboQuant_mse algorithm does NOT show normalization:
```
y <- Pi * x
idx_j <- argmin_k |y_j - c_k|
```
This suggests PolarQuant handles norm separation implicitly, or the algorithm assumes inputs are already on a sphere.

### 3. "No per-block full-precision constants"
The website explicitly says overhead drops to zero. Our implementation stores 3 fp16 values per block (norm, rnorm_hi, rnorm_lo) = 6 bytes overhead per 128 elements.

### 4. QJL dequant formula
Website: `r_hat = sqrt(pi/2d) * S^T * Q_qjl(r)`
Our code: `corr[j] *= sqrt(pi/2) / m * rnorm`
The website formula doesn't explicitly show rnorm multiplication — it may be implicit or handled differently in the full pipeline.

### 5. Residual computed in original space
TurboQuant_prod step 3.2: `r <- x - x_bar_mse` — residual in original space after full MSE dequant. Our implementation computes residual in normalized rotated space.

### 6. Variance bound includes ||y||^2 and ||r||^2
`D_prod <= (pi^2*sqrt(3) * ||y||^2/d) * 1/4^b`
This scales with both query norm and residual norm, divided by dimension d.
