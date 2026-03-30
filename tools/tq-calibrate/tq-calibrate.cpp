// TurboQuant offline calibration tool
// Runs a forward pass on calibration text, accumulates per-head K channel magnitudes,
// computes per-layer-per-head channel permutations (outlier detection), and saves them.
//
// Usage: llama-tq-calibrate -m model.gguf -f calibration.txt -o perms.bin [--pre-rope]
//
// The output binary can be injected into a GGUF file as blk.{layer}.tq_k_channel_perm tensors.

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// Per-layer-per-head accumulator for channel magnitudes
struct tq_calibration_data {
    int n_layers   = 0;
    int n_kv_heads = 0;
    int head_dim   = 0;
    bool pre_rope  = false;

    // [n_layers][n_kv_heads][head_dim] — accumulated |K| per channel
    std::vector<double> accum;
    // [n_layers] — token count per layer
    std::vector<int64_t> counts;

    // Map from model layer index to calibration layer index (only KV layers)
    std::vector<int> layer_map; // model_il -> calib_idx, or -1

    std::mutex mtx;

    void init(int n_layer_model, int n_kv_heads_, int head_dim_, bool pre_rope_,
              const std::function<bool(int)> & has_kv) {
        n_kv_heads = n_kv_heads_;
        head_dim   = head_dim_;
        pre_rope   = pre_rope_;

        layer_map.resize(n_layer_model, -1);
        int idx = 0;
        for (int il = 0; il < n_layer_model; il++) {
            if (has_kv(il)) {
                layer_map[il] = idx++;
            }
        }
        n_layers = idx;

        accum.resize((size_t)n_layers * n_kv_heads * head_dim, 0.0);
        counts.resize(n_layers, 0);
    }

    double * get_accum(int calib_layer, int head) {
        return accum.data() + ((size_t)calib_layer * n_kv_heads + head) * head_dim;
    }

    void accumulate(int model_il, const float * data, int64_t n_embd_k_gqa, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;

        std::lock_guard<std::mutex> lock(mtx);

        if (pre_rope) {
            // Pre-RoPE: data is [n_embd_k_gqa, n_tokens], interleaved heads
            // Split into heads manually
            for (int64_t t = 0; t < n_tokens; t++) {
                const float * row = data + t * n_embd_k_gqa;
                for (int h = 0; h < n_kv_heads; h++) {
                    double * acc = get_accum(cidx, h);
                    const float * head_data = row + h * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += fabs((double)head_data[d]);
                    }
                }
            }
        } else {
            // Post-RoPE: data is [head_dim, n_kv_heads, n_tokens] (3D reshaped)
            for (int64_t t = 0; t < n_tokens; t++) {
                for (int h = 0; h < n_kv_heads; h++) {
                    double * acc = get_accum(cidx, h);
                    const float * head_data = data + (t * n_kv_heads + h) * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += fabs((double)head_data[d]);
                    }
                }
            }
        }
        counts[cidx] += n_tokens;
    }

    // Compute permutation for one (layer, head): sort channels by mean |K| descending
    // First n_hi entries = outlier channels, rest = regular
    void compute_perm(int calib_layer, int head, uint8_t * perm) const {
        const double * acc = accum.data() + ((size_t)calib_layer * n_kv_heads + head) * head_dim;
        int64_t cnt = counts[calib_layer];
        if (cnt == 0) cnt = 1;

        // Compute mean magnitude per channel
        std::vector<std::pair<double, int>> mags(head_dim);
        for (int d = 0; d < head_dim; d++) {
            mags[d] = { acc[d] / (double)cnt, d };
        }

        // Sort descending by magnitude
        std::sort(mags.begin(), mags.end(), [](const auto & a, const auto & b) {
            return a.first > b.first;
        });

        // Permutation: outliers first (highest magnitude), then regulars
        for (int d = 0; d < head_dim; d++) {
            perm[d] = (uint8_t)mags[d].second;
        }
    }
};

static tq_calibration_data g_calib;
static std::vector<char> g_tensor_buf; // reusable buffer for GPU readback

static bool tq_collect_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    // We want to intercept Kcur tensors
    // The tensor passed to cb_eval has src[0] = the actual data tensor
    // But actually, cb_eval is called on the result tensor of the op that has the name

    // For the "ask" phase: we want tensors named "Kcur"
    if (ask) {
        // Match tensors named "Kcur-{il}"
        int il_tmp;
        if (sscanf(t->name, "Kcur-%d", &il_tmp) == 1) {
            // Pre-RoPE mode: capture 2D tensors (before reshape)
            // Post-RoPE mode: capture 3D tensors (after reshape + rope)
            if (g_calib.pre_rope) {
                return (ggml_n_dims(t) == 2); // [n_embd_k_gqa, n_tokens]
            } else {
                return (ggml_n_dims(t) == 3); // [head_dim, n_kv_heads, n_tokens]
            }
        }
        return false;
    }

    // Collect phase: read data and accumulate
    // Tensor name format: "Kcur-{layer_index}" (set by llama_context::graph_get_cb)
    int il = -1;
    if (sscanf(t->name, "Kcur-%d", &il) != 1) return true;
    if (il < 0 || il >= (int)g_calib.layer_map.size()) return true;
    if (g_calib.layer_map[il] < 0) return true;

    // Read tensor data (may be on GPU)
    const size_t nbytes = ggml_nbytes(t);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    const float * data;
    if (is_host) {
        data = (const float *)t->data;
    } else {
        if (g_tensor_buf.size() < nbytes) {
            g_tensor_buf.resize(nbytes);
        }
        ggml_backend_tensor_get(t, g_tensor_buf.data(), 0, nbytes);
        data = (const float *)g_tensor_buf.data();
    }

    if (t->type != GGML_TYPE_F32) return true; // only handle f32

    if (g_calib.pre_rope) {
        // 2D: [n_embd_k_gqa, n_tokens]
        g_calib.accumulate(il, data, t->ne[0], t->ne[1]);
    } else {
        // 3D: [head_dim, n_kv_heads, n_tokens]
        g_calib.accumulate(il, data, t->ne[0] * t->ne[1], t->ne[2]);
    }

    return true;
}

static std::vector<llama_token> tokenize_file(const llama_vocab * vocab, const std::string & path, int max_tokens) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "error: failed to open '%s'\n", path.c_str());
        return {};
    }
    std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    auto tokens = common_tokenize(vocab, text, true);
    if (max_tokens > 0 && (int)tokens.size() > max_tokens) {
        tokens.resize(max_tokens);
    }
    return tokens;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m model.gguf -f calibration.txt -o perms.bin [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -m,  --model FILE      GGUF model file\n");
    fprintf(stderr, "  -f,  --file FILE       Calibration text file\n");
    fprintf(stderr, "  -o,  --output FILE     Output permutation file (default: tq-perms.bin)\n");
    fprintf(stderr, "  -n,  --n-tokens N      Max calibration tokens (default: 4096)\n");
    fprintf(stderr, "  -c,  --ctx-size N      Context size (default: 512)\n");
    fprintf(stderr, "  -ngl,--n-gpu-layers N  GPU layers (default: 99)\n");
    fprintf(stderr, "       --pre-rope        Capture pre-RoPE K (default: post-RoPE)\n");
    fprintf(stderr, "       --post-rope       Capture post-RoPE K (default)\n");
    fprintf(stderr, "\nOutput format: binary file with header + per-layer-per-head permutations.\n");
    fprintf(stderr, "Use llama-gguf-inject (TODO) to add permutations to the GGUF model.\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string calib_file;
    std::string output_path = "tq-perms.bin";
    int n_tokens_max = 4096;
    int n_ctx = 512;
    int n_gpu_layers = 99;
    bool pre_rope = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            calib_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-n" || arg == "--n-tokens") && i + 1 < argc) {
            n_tokens_max = std::atoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            n_ctx = std::atoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--pre-rope") {
            pre_rope = true;
        } else if (arg == "--post-rope") {
            pre_rope = false;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || calib_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Get model info
    const int n_layer    = llama_model_n_layer(model);
    const int n_kv_heads = llama_model_n_head_kv(model);
    const int head_dim   = llama_model_n_embd(model) / llama_model_n_head(model);

    fprintf(stderr, "model: %d layers, %d KV heads, head_dim=%d\n", n_layer, n_kv_heads, head_dim);

    // Determine which layers have KV (for now assume all — Qwen3.5 SWA handling TODO via hparams)
    // The has_kv check happens at KV cache level; for calibration we collect all layers
    // and the KV cache constructor will filter. We use a simple heuristic:
    // all layers have KV unless the model has n_layer_kv_from_start set.
    auto has_kv = [&](int /*il*/) -> bool { return true; };

    g_calib.init(n_layer, n_kv_heads, head_dim, pre_rope, has_kv);

    fprintf(stderr, "calibration: %d KV layers, %s mode, max %d tokens\n",
            g_calib.n_layers, pre_rope ? "pre-RoPE" : "post-RoPE", n_tokens_max);

    // Tokenize calibration file
    auto tokens = tokenize_file(vocab, calib_file, n_tokens_max);
    if (tokens.empty()) {
        fprintf(stderr, "error: no tokens from calibration file\n");
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "tokenized %zu tokens from '%s'\n", tokens.size(), calib_file.c_str());

    // Create context with f16 K cache (no TQ quantization during calibration)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.type_k   = GGML_TYPE_F16;
    cparams.type_v   = GGML_TYPE_F16;
    cparams.cb_eval  = tq_collect_callback;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Process tokens in chunks
    const int n_tokens_total = (int)tokens.size();
    int n_processed = 0;

    while (n_processed < n_tokens_total) {
        int n_batch = std::min(n_ctx, n_tokens_total - n_processed);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int i = 0; i < n_batch; i++) {
            common_batch_add(batch, tokens[n_processed + i], n_processed + i, {0}, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "warning: decode failed at position %d\n", n_processed);
        }

        n_processed += n_batch;
        fprintf(stderr, "\r  processed %d / %d tokens", n_processed, n_tokens_total);

        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    fprintf(stderr, "\n");

    // Compute permutations
    fprintf(stderr, "computing channel permutations...\n");

    const int n_hi = head_dim / 4; // outlier count
    std::vector<uint8_t> all_perms((size_t)g_calib.n_layers * n_kv_heads * head_dim);

    for (int l = 0; l < g_calib.n_layers; l++) {
        for (int h = 0; h < n_kv_heads; h++) {
            uint8_t * perm = all_perms.data() + ((size_t)l * n_kv_heads + h) * head_dim;
            g_calib.compute_perm(l, h, perm);
        }

        if (g_calib.counts[l] > 0) {
            fprintf(stderr, "  layer %2d: %lld tokens accumulated\n", l, (long long)g_calib.counts[l]);
        }
    }

    // Save permutation file
    // Format: [magic:4][version:4][n_layers:4][n_kv_heads:4][head_dim:4][pre_rope:4]
    //         [layer_map: n_layer_model * 4 bytes]
    //         [perms: n_layers * n_kv_heads * head_dim bytes]
    {
        FILE * fp = fopen(output_path.c_str(), "wb");
        if (!fp) {
            fprintf(stderr, "error: failed to open output '%s'\n", output_path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        uint32_t magic   = 0x54515045; // "TQPE" (TQ PErmutation)
        uint32_t version = 1;
        uint32_t nl      = (uint32_t)g_calib.n_layers;
        uint32_t nh      = (uint32_t)n_kv_heads;
        uint32_t hd      = (uint32_t)head_dim;
        uint32_t pr      = pre_rope ? 1 : 0;
        uint32_t nlm     = (uint32_t)n_layer;

        fwrite(&magic,   4, 1, fp);
        fwrite(&version, 4, 1, fp);
        fwrite(&nl,      4, 1, fp);
        fwrite(&nh,      4, 1, fp);
        fwrite(&hd,      4, 1, fp);
        fwrite(&pr,      4, 1, fp);
        fwrite(&nlm,     4, 1, fp);

        // Write layer map (model_il -> calib_idx)
        for (int il = 0; il < n_layer; il++) {
            int32_t idx = (int32_t)g_calib.layer_map[il];
            fwrite(&idx, 4, 1, fp);
        }

        // Write permutations
        fwrite(all_perms.data(), 1, all_perms.size(), fp);

        fclose(fp);

        fprintf(stderr, "saved %zu bytes to '%s'\n",
                28 + n_layer * 4 + all_perms.size(), output_path.c_str());
        fprintf(stderr, "  %d KV layers × %d heads × %d channels = %zu permutation entries\n",
                g_calib.n_layers, n_kv_heads, head_dim, all_perms.size());
        fprintf(stderr, "  outlier channels per head: %d (top %.0f%% by magnitude)\n",
                n_hi, 100.0 * n_hi / head_dim);
    }

    // Print top outlier channels for first layer/head as sanity check
    {
        fprintf(stderr, "\nsample: layer 0, head 0 — top %d outlier channels:\n  ", n_hi);
        const uint8_t * perm = all_perms.data();
        for (int i = 0; i < n_hi && i < 16; i++) {
            fprintf(stderr, "%d ", perm[i]);
        }
        if (n_hi > 16) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
