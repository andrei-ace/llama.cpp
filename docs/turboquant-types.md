# TurboQuant Type Reference

All TurboQuant KV cache quantization types, their exact bits-per-value (bpv), and
block structure. Computed from the struct definitions in `ggml/src/ggml-common.h`.

Formula: **bpv = sizeof(block) * 8 / block_size**

## Naming Convention

| Suffix | Meaning |
|--------|---------|
| `_0`   | No QJL correction (MSE quantization only) |
| `_0j`  | Uniform (no split) + QJL on full vector |
| `_1j`  | Uniform (no split) + QJL on residual (lower MSE bits) |
| `_s`   | Split (channel separation into outlier/regular subsets) |
| `_sj`  | Split + QJL signs stored for both subsets; QJL correction applied to hi only during decode |
| `_sjj` | Split + QJL correction applied to both hi and lo subsets during decode |

**Split types** separate each head's channels into 25% outlier ("hi") and 75% regular
("lo") subsets based on calibrated activation magnitudes. Each subset gets independent
rotation, norms, MSE quantization, and QJL sign bits. All split types require
calibration via `llama-tq-calibrate`.

## d=128 Types

Constants: `TQK_BLOCK_SIZE = 128`, `TQK_N_OUTLIER = 32`, `TQK_N_REGULAR = 96`

### Uniform types (no channel split)

| CLI name | Internal struct | Bytes | bpv | Description | Calibration |
|----------|----------------|-------|-----|-------------|-------------|
| `tqk4_0` | `block_tqk_had_mse4` | 66 | 4.125 | H_128 Hadamard + 4-bit MSE | No |
| `tqk5_0j` | `block_tqk_had_prod5` | 84 | 5.25 | 4-bit MSE + 1-bit QJL on residual | No |
| `tqk4_1j` | `block_tqk_had_prod4` | 68 | 4.25 | 3-bit MSE + 1-bit QJL on residual | No |
| `tqv4_0` | `block_tqv_had_mse4` | 66 | 4.125 | V cache: 4-bit MSE (same struct as tqk4_0) | No |

### Split types (32/96 outlier/regular channel split)

| CLI name | Internal struct | Bytes | bpv | Hi (32ch) | Lo (96ch) | Calibration |
|----------|----------------|-------|-----|-----------|-----------|-------------|
| `tqk3_sj` | `block_tqk_5hi_3lo` | 62 | 3.875 | 4-bit MSE + QJL hi | 3-bit MSE | Yes |
| `tqk4_sj` | `block_tqk_6hi_3lo` | 66 | 4.125 | 5-bit MSE + QJL hi | 3-bit MSE | Yes |
| `tqk4_sjj` | `block_tqk_6hi_3lo_jj` | 80 | 5.00 | 5-bit MSE + QJL | 3-bit MSE + QJL | Yes |
| `tqk3_sjj` | `block_tqk_3hi_2lo` | 60 | 3.75 | 3-bit MSE + QJL | 2-bit MSE + QJL | Yes |
| `tqk2_sjjj` | `block_tqk_2hi_1lo` | 44 | 2.75 | 2-bit MSE + QJL | 1-bit MSE + QJL | Yes |

> **Note on tqk4_sj vs tqk4_sjj:** `tqk4_sj` uses `block_tqk_6hi_3lo` (66 bytes, 4.125 bpv)
> with QJL on hi only. `tqk4_sjj` uses `block_tqk_6hi_3lo_jj` (80 bytes, 5.00 bpv)
> with QJL on both hi and lo channels. They have different structs.
>
> **Note on tqk3_sjj:** Previously named `tqk3_sjj` in documentation and benchmarks.
> The CLI name is now `tqk3_sjj`. Uses `block_tqk_3hi_2lo`.

## d=256 Types

Constants: `TQK_BLOCK_SIZE_D256 = 256`, `TQK_N_OUTLIER_D256 = 64`, `TQK_N_REGULAR_D256 = 192`

### Uniform types (no channel split)

| CLI name | Internal struct | Bytes | bpv | Description | Calibration |
|----------|----------------|-------|-----|-------------|-------------|
| `tqk4_0_d256` | `block_tqk_had_mse4_d256` | 130 | 4.0625 | H_256 Hadamard + 4-bit MSE | No |
| `tqk5_0_d256` | `block_tqk_had_prod5_d256` | 164 | 5.125 | 4-bit MSE + 1-bit QJL | No |
| `tqk4_1_d256` | `block_tqk_had_prod4_d256` | 132 | 4.125 | 3-bit MSE + 1-bit QJL | No |
| `tqv4_0_d256` | `block_tqv_had_mse4_d256` | 130 | 4.0625 | V cache d=256 (same struct as tqk4_0_d256) | No |

### Split types (64/192 outlier/regular channel split)

| CLI name | Internal struct | Bytes | bpv | Hi (64ch) | Lo (192ch) | Calibration |
|----------|----------------|-------|-----|-----------|------------|-------------|
| `tqk3_sj_d256` | `block_tqk_5hi_3lo_d256` | 144 | 4.50 | 4-bit MSE + QJL | 3-bit MSE + QJL | Yes |
| `tqk4_sj_d256` | `block_tqk_6hi_3lo_d256` | 152 | 4.75 | 5-bit MSE + QJL | 3-bit MSE + QJL | Yes |
| `tqk4_sjj_d256` | `block_tqk_6hi_3lo_d256` | 152 | 4.75 | 5-bit MSE + QJL | 3-bit MSE + QJL | Yes |
| `tqk3_sjj_d256` | `block_tqk_3hi_2lo_d256` | 112 | 3.50 | 3-bit MSE + QJL | 2-bit MSE + QJL | Yes |
| `tqk2_sjj_d256` | `block_tqk_2hi_1lo_d256` | 80 | 2.50 | 2-bit MSE + QJL | 1-bit MSE + QJL | Yes |

## d=128 vs d=256 bpv Comparison

| Type family | d=128 bpv | d=256 bpv | Savings |
|-------------|-----------|-----------|---------|
| tqk4_0 | 4.125 | 4.0625 | 1.5% |
| tqk5_0j | 5.25 | 5.125 | 2.4% |
| tqk4_1j | 4.25 | 4.125 | 2.9% |
| tqk3_sj | 4.75 | 4.50 | 5.3% |
| tqk4_sj | 5.00 | 4.75 | 5.0% |
| tqk3_sjj | 3.75 | 3.50 | 6.7% |
| tqk2_sjj | 2.75 | 2.50 | 9.1% |

> d=256 types amortize the fixed per-block overhead (norms, rnorms) over twice as many
> elements, yielding lower bpv. The savings are proportionally larger for types with
> more overhead fields (split types have 4 half-precision norms = 8 bytes overhead).

## Block Layout Details

### block_tqk_had_mse4 (tqk4_0, 66 bytes, d=128)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm of rotated block |
| `qs[64]` | 64B | 4-bit MSE indices (128 channels, 2 per byte) |

### block_tqk_had_prod5 (tqk5_0j, 84 bytes, d=128)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm of rotated block |
| `rnorm` | 2B (ggml_half) | QJL residual norm |
| `qs[64]` | 64B | 4-bit MSE indices (128 channels) |
| `signs[16]` | 16B | 1-bit QJL signs (128 channels) |

### block_tqk_had_prod4 (tqk4_1j, 68 bytes, d=128)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm of rotated block |
| `rnorm` | 2B (ggml_half) | QJL residual norm |
| `qs[48]` | 48B | 3-bit MSE indices (128 channels) |
| `signs[16]` | 16B | 1-bit QJL signs (128 channels) |

### block_tqk_5hi_3lo (tqk3_sj, 62 bytes, d=128)

QJL on hi only — no rnorm_lo/signs_lo fields.

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `qs_hi[16]` | 16B | 4-bit MSE indices (32 outlier channels) |
| `qs_lo[36]` | 36B | 3-bit MSE indices (96 regular channels) |
| `signs_hi[4]` | 4B | 1-bit QJL signs (32 outlier channels) |

### block_tqk_6hi_3lo (tqk4_sj, 66 bytes, d=128)

QJL on hi only — no rnorm_lo/signs_lo fields.

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `qs_hi[20]` | 20B | 5-bit MSE indices (32 outlier channels) |
| `qs_lo[36]` | 36B | 3-bit MSE indices (96 regular channels) |
| `signs_hi[4]` | 4B | 1-bit QJL signs (32 outlier channels) |

### block_tqk_6hi_3lo_jj (tqk4_sjj, 80 bytes, d=128)

QJL on both hi and lo channels.

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[20]` | 20B | 5-bit MSE indices (32 outlier channels) |
| `qs_lo[36]` | 36B | 3-bit MSE indices (96 regular channels) |
| `signs_hi[4]` | 4B | 1-bit QJL signs (32 outlier channels) |
| `signs_lo[12]` | 12B | 1-bit QJL signs (96 regular channels) |

### block_tqk_3hi_2lo (tqk3_sjj, 60 bytes, d=128)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[12]` | 12B | 3-bit MSE indices (32 outlier channels) |
| `qs_lo[24]` | 24B | 2-bit MSE indices (96 regular channels) |
| `signs_hi[4]` | 4B | 1-bit QJL signs (32 outlier channels) |
| `signs_lo[12]` | 12B | 1-bit QJL signs (96 regular channels) |

### block_tqk_2hi_1lo (tqk2_sjj, 44 bytes, d=128)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[8]` | 8B | 2-bit MSE indices (32 outlier channels) |
| `qs_lo[12]` | 12B | 1-bit MSE indices (96 regular channels) |
| `signs_hi[4]` | 4B | 1-bit QJL signs (32 outlier channels) |
| `signs_lo[12]` | 12B | 1-bit QJL signs (96 regular channels) |

### block_tqk_had_mse4_d256 (tqk4_0_d256, 130 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm of rotated block |
| `qs[128]` | 128B | 4-bit MSE indices (256 channels) |

### block_tqk_had_prod5_d256 (tqk5_0_d256, 164 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm |
| `rnorm` | 2B (ggml_half) | QJL residual norm |
| `qs[128]` | 128B | 4-bit MSE indices (256 channels) |
| `signs[32]` | 32B | 1-bit QJL signs (256 channels) |

### block_tqk_had_prod4_d256 (tqk4_1_d256, 132 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm` | 2B (ggml_half) | L2 norm |
| `rnorm` | 2B (ggml_half) | QJL residual norm |
| `qs[96]` | 96B | 3-bit MSE indices (256 channels) |
| `signs[32]` | 32B | 1-bit QJL signs (256 channels) |

### block_tqk_5hi_3lo_d256 (tqk3_sj_d256, 144 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[32]` | 32B | 4-bit MSE indices (64 outlier channels) |
| `qs_lo[72]` | 72B | 3-bit MSE indices (192 regular channels) |
| `signs_hi[8]` | 8B | 1-bit QJL signs (64 outlier channels) |
| `signs_lo[24]` | 24B | 1-bit QJL signs (192 regular channels) |

### block_tqk_6hi_3lo_d256 (tqk4_sj_d256 / tqk4_sjj_d256, 152 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[40]` | 40B | 5-bit MSE indices (64 outlier channels) |
| `qs_lo[72]` | 72B | 3-bit MSE indices (192 regular channels) |
| `signs_hi[8]` | 8B | 1-bit QJL signs (64 outlier channels) |
| `signs_lo[24]` | 24B | 1-bit QJL signs (192 regular channels) |

### block_tqk_3hi_2lo_d256 (tqk3_sjj_d256, 112 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[24]` | 24B | 3-bit MSE indices (64 outlier channels) |
| `qs_lo[48]` | 48B | 2-bit MSE indices (192 regular channels) |
| `signs_hi[8]` | 8B | 1-bit QJL signs (64 outlier channels) |
| `signs_lo[24]` | 24B | 1-bit QJL signs (192 regular channels) |

### block_tqk_2hi_1lo_d256 (tqk2_sjj_d256, 80 bytes, d=256)

| Field | Size | Purpose |
|-------|------|---------|
| `norm_hi` | 2B (ggml_half) | Outlier subset L2 norm |
| `norm_lo` | 2B (ggml_half) | Regular subset L2 norm |
| `rnorm_hi` | 2B (ggml_half) | Outlier QJL residual norm |
| `rnorm_lo` | 2B (ggml_half) | Regular QJL residual norm |
| `qs_hi[16]` | 16B | 2-bit MSE indices (64 outlier channels) |
| `qs_lo[24]` | 24B | 1-bit MSE indices (192 regular channels) |
| `signs_hi[8]` | 8B | 1-bit QJL signs (64 outlier channels) |
| `signs_lo[24]` | 24B | 1-bit QJL signs (192 regular channels) |

## GGML Type Enum Mapping

For programmatic reference, the mapping between CLI names and `ggml_type` enum values
(defined in `ggml/include/ggml.h`):

| CLI name | Enum | Value |
|----------|------|-------|
| `tqk3_sj` | `GGML_TYPE_TQK_5HI_3LO_HAD` | 50 |
| `tqk4_0` | `GGML_TYPE_TQK_HAD_MSE4` | 51 |
| `tqk5_0j` | `GGML_TYPE_TQK_HAD_PROD5` | 52 |
| `tqk4_1j` | `GGML_TYPE_TQK_HAD_PROD4` | 53 |
| `tqv4_0` | `GGML_TYPE_TQV_HAD_MSE4` | 54 |
| `tqk4_0_d256` | `GGML_TYPE_TQK_HAD_MSE4_D256` | 55 |
| `tqk5_0_d256` | `GGML_TYPE_TQK_HAD_PROD5_D256` | 56 |
| `tqk4_1_d256` | `GGML_TYPE_TQK_HAD_PROD4_D256` | 57 |
| `tqk3_sj_d256` | `GGML_TYPE_TQK_5HI_3LO_HAD_D256` | 58 |
| `tqv4_0_d256` | `GGML_TYPE_TQV_HAD_MSE4_D256` | 59 |
| `tqk4_sj` | `GGML_TYPE_TQK_6HI_3LO_HAD` | 60 |
| `tqk2_sjj` | `GGML_TYPE_TQK_2HI_1LO_HAD` | 61 |
| `tqk3_sjj` | `GGML_TYPE_TQK_3HI_2LO_HAD` | 62 |
| `tqk4_sj_d256` | `GGML_TYPE_TQK_6HI_3LO_HAD_D256` | 63 |
| `tqk2_sjj_d256` | `GGML_TYPE_TQK_2HI_1LO_HAD_D256` | 64 |
| `tqk3_sjj_d256` | `GGML_TYPE_TQK_3HI_2LO_HAD_D256` | 65 |
| `tqk4_sjj` | `GGML_TYPE_TQK_6HI_3LO_HAD_JJ` | 66 |
| `tqk4_sjj_d256` | `GGML_TYPE_TQK_6HI_3LO_HAD_JJ_D256` | 67 |

## Name Migration

| Old name (docs/benchmarks) | Current CLI name | Notes |
|---------------------------|------------------|-------|
| `tqk3_sjj` | `tqk3_sjj` | Renamed to reflect QJL on both subsets |
| `tqk3_0j` | `tqk3_sj` | Header comment says `tqk3_0j` but registered type_name is `tqk3_sj` |
| `tqk5_0j_d256` | `tqk5_0_d256` | d=256 uniform types drop the `j` suffix in type_name |
| `tqk4_1j_d256` | `tqk4_1_d256` | d=256 uniform types drop the `j` suffix in type_name |
