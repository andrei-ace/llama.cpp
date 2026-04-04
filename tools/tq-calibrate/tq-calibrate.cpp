// TurboQuant offline calibration tool
// Accumulates per-head K channel variance, computes per-layer-per-head
// channel permutations (sorted by importance), and saves them.
//
// Usage: llama-tq-calibrate -m model.gguf -f calibration.txt -o perms.bin [options]
//
// The permutation file stores channels sorted by importance (highest first).
// At runtime, TQL uses: top 32 = hi (q8), next 32 = mid (3mse+qjl), bottom 64 = low (2mse+qjl).
// TQ3J/TQ2J use identity (no permutation needed — full FWHT-128).

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

enum calib_metric_t { METRIC_MAG = 0, METRIC_VAR = 1, METRIC_BOTH = 2 };

struct tq_calibration_data {
    int n_layers   = 0;
    int max_head_dim = 0;
    bool pre_rope  = false;
    calib_metric_t metric = METRIC_VAR;

    // per-layer dimensions (for models with mixed head dims like Gemma4)
    std::vector<int> layer_head_dim;   // head_dim per calibration layer
    std::vector<int> layer_n_kv_heads; // n_kv_heads per calibration layer

    // [n_layers][max_n_kv_heads][max_head_dim] — padded to max dims
    int max_n_kv_heads = 0;
    std::vector<double> sum_abs;
    std::vector<double> sum_val;
    std::vector<double> sum_sq;
    std::vector<int64_t> counts;
    std::vector<int> layer_map;
    std::mutex mtx;

    void init(int n_layer_model, const std::vector<int> & kv_heads_per_layer,
              const std::vector<int> & head_dim_per_layer, bool pre_rope_,
              calib_metric_t metric_) {
        pre_rope   = pre_rope_;
        metric     = metric_;

        layer_map.resize(n_layer_model, -1);
        int idx = 0;
        for (int il = 0; il < n_layer_model; il++) {
            layer_map[il] = idx++;
        }
        n_layers = idx;

        layer_head_dim.resize(n_layers);
        layer_n_kv_heads.resize(n_layers);
        max_head_dim = 0;
        max_n_kv_heads = 0;
        for (int il = 0; il < n_layer_model; il++) {
            int cidx = layer_map[il];
            if (cidx >= 0) {
                layer_head_dim[cidx]   = head_dim_per_layer[il];
                layer_n_kv_heads[cidx] = kv_heads_per_layer[il];
                if (head_dim_per_layer[il] > max_head_dim) max_head_dim = head_dim_per_layer[il];
                if (kv_heads_per_layer[il] > max_n_kv_heads) max_n_kv_heads = kv_heads_per_layer[il];
            }
        }

        size_t sz = (size_t)n_layers * max_n_kv_heads * max_head_dim;
        sum_abs.resize(sz, 0.0);
        sum_val.resize(sz, 0.0);
        sum_sq.resize(sz, 0.0);
        counts.resize(n_layers, 0);
    }

    void accum_head(int cidx, int h, const float * hd) {
        int hd_dim = layer_head_dim[cidx];
        size_t off = ((size_t)cidx * max_n_kv_heads + h) * max_head_dim;
        double * sa = sum_abs.data() + off;
        double * sv = sum_val.data() + off;
        double * ss = sum_sq.data()  + off;
        for (int d = 0; d < hd_dim; d++) {
            double v = (double)hd[d];
            sa[d] += fabs(v);
            sv[d] += v;
            ss[d] += v * v;
        }
    }

    void accumulate(int model_il, const float * data, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;
        int n_heads = layer_n_kv_heads[cidx];
        int hdim    = layer_head_dim[cidx];
        std::lock_guard<std::mutex> lock(mtx);
        for (int64_t t = 0; t < n_tokens; t++)
            for (int h = 0; h < n_heads; h++)
                accum_head(cidx, h, data + (t * n_heads + h) * hdim);
        counts[cidx] += n_tokens;
    }

    void accumulate_2d(int model_il, const float * data, int64_t n_embd_k_gqa, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;
        int n_heads = layer_n_kv_heads[cidx];
        int hdim    = layer_head_dim[cidx];
        std::lock_guard<std::mutex> lock(mtx);
        for (int64_t t = 0; t < n_tokens; t++)
            for (int h = 0; h < n_heads; h++)
                accum_head(cidx, h, data + t * n_embd_k_gqa + h * hdim);
        counts[cidx] += n_tokens;
    }

    void compute_perm(int cidx, int head, uint8_t * perm) const {
        int hd_dim = layer_head_dim[cidx];
        size_t off = ((size_t)cidx * max_n_kv_heads + head) * max_head_dim;
        const double * sa = sum_abs.data() + off;
        const double * sv = sum_val.data() + off;
        const double * ss = sum_sq.data()  + off;
        int64_t n = std::max(counts[cidx], (int64_t)1);

        std::vector<std::pair<double, int>> importance(hd_dim);
        for (int d = 0; d < hd_dim; d++) {
            double mag = sa[d] / n;
            double var = ss[d] / n - (sv[d] / n) * (sv[d] / n);
            if (var < 0) var = 0;

            double score;
            if (metric == METRIC_MAG)       score = mag;
            else if (metric == METRIC_VAR)  score = var;
            else /* METRIC_BOTH */          score = mag * sqrt(var);
            importance[d] = { score, d };
        }

        std::sort(importance.begin(), importance.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int d = 0; d < hd_dim; d++) perm[d] = (uint8_t)importance[d].second;
    }
};

static tq_calibration_data g_calib;
static std::vector<char> g_tensor_buf;

static bool tq_collect_callback(struct ggml_tensor * t, bool ask, void *) {
    if (ask) {
        // Capture Kcur and Kcur_normed (iSWA global layers only emit Kcur_normed)
        int il_tmp;
        if (sscanf(t->name, "Kcur-%d", &il_tmp) == 1) return true;
        if (sscanf(t->name, "Kcur_normed-%d", &il_tmp) == 1) return true;
        return false;
    }

    // Debug: log all tensors that contain layer 5 (a global layer)
    {
        int tmp_il = -1;
        if (sscanf(t->name, "%*[^-]-%d", &tmp_il) == 1 && (tmp_il == 5 || tmp_il == 11)) {
            static int dbg_count = 0;
            if (dbg_count < 50) {
                fprintf(stderr, "  cb_data: il=%d name='%s' shape=[%lld,%lld,%lld] type=%s\n",
                        tmp_il, t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                        ggml_type_name(t->type));
                dbg_count++;
            }
        }
    }

    if (t->type != GGML_TYPE_F32) return true;
    int il = -1;
    if (sscanf(t->name, "Kcur-%d", &il) != 1 &&
        sscanf(t->name, "Kcur_normed-%d", &il) != 1) return true;
    // Prefer Kcur over Kcur_normed (skip normed if we already have raw for this layer)
    if (strstr(t->name, "_normed") && il >= 0 && il < (int)g_calib.layer_map.size()) {
        int cidx = g_calib.layer_map[il];
        if (cidx >= 0 && g_calib.counts[cidx] > 0) return true; // already have raw Kcur
    }

    // Debug: print all captured Kcur tensors
    static bool debug_printed[256] = {};
    if (il >= 0 && il < 256 && !debug_printed[il]) {
        fprintf(stderr, "  cb: Kcur-%d shape=[%lld,%lld,%lld,%lld] type=%s\n",
                il, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                ggml_type_name(t->type));
        debug_printed[il] = true;
    }

    if (il < 0 || il >= (int)g_calib.layer_map.size() || g_calib.layer_map[il] < 0) return true;

    const size_t nbytes = ggml_nbytes(t);
    const float * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const float *)t->data;
    } else {
        if (g_tensor_buf.size() < nbytes) g_tensor_buf.resize(nbytes);
        ggml_backend_tensor_get(t, g_tensor_buf.data(), 0, nbytes);
        data = (const float *)g_tensor_buf.data();
    }

    // Auto-detect per-layer dimensions from the actual tensor shape
    int cidx = g_calib.layer_map[il];
    if (ggml_n_dims(t) >= 3) {
        // 3D: [head_dim, n_heads, n_tokens]
        int actual_hdim = (int)t->ne[0];
        int actual_nkv  = (int)t->ne[1];
        if (g_calib.layer_head_dim[cidx] != actual_hdim || g_calib.layer_n_kv_heads[cidx] != actual_nkv) {
            g_calib.layer_head_dim[cidx]   = actual_hdim;
            g_calib.layer_n_kv_heads[cidx] = actual_nkv;
            if (actual_hdim > g_calib.max_head_dim) { g_calib.max_head_dim = actual_hdim; }
            if (actual_nkv > g_calib.max_n_kv_heads) { g_calib.max_n_kv_heads = actual_nkv; }
        }
    } else {
        // 2D: [n_embd_k_gqa, n_tokens] — infer from n_embd_k_gqa / n_kv_heads
        // (can't auto-detect head_dim from 2D without knowing n_kv_heads)
    }

    if (ggml_n_dims(t) == 2) {
        g_calib.accumulate_2d(il, data, t->ne[0], t->ne[1]);
    } else {
        g_calib.accumulate(il, data, t->ne[2]);
    }
    return true;
}

static std::vector<llama_token> tokenize_file(const llama_vocab * vocab, const std::string & path, int max_tokens) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) { fprintf(stderr, "error: failed to open '%s'\n", path.c_str()); return {}; }
    std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto tokens = common_tokenize(vocab, text, true);
    if (max_tokens > 0 && (int)tokens.size() > max_tokens) tokens.resize(max_tokens);
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
    fprintf(stderr, "       --pre-rope        Capture pre-RoPE K (default)\n");
    fprintf(stderr, "       --post-rope       Capture post-RoPE K\n");
    fprintf(stderr, "       --metric var      Outlier = highest variance Var(K) (default)\n");
    fprintf(stderr, "       --metric mag      Outlier = highest mean |K|\n");
    fprintf(stderr, "       --metric both     Outlier = |K| x std(K)\n");
    fprintf(stderr, "       --stats FILE      Dump per-channel stats CSV\n");
    fprintf(stderr, "       --threshold N     Override layers with outlier%% > N to --override-type (default: 101 = none)\n");
    fprintf(stderr, "       --override-type T Override type for extreme layers: f16, q8_0, q4_0 (default: q8_0)\n");
    fprintf(stderr, "\nOutput format: channels sorted by importance (most important first).\n");
    fprintf(stderr, "TQL uses top 32 = hi (q8), next 32 = mid (3mse+qjl), bottom 64 = low (2mse+qjl).\n");
}

int main(int argc, char ** argv) {
    std::string model_path, calib_file, output_path = "tq-perms.bin", stats_path;
    int n_tokens_max = 4096, n_ctx = 512, n_gpu_layers = 99;
    bool pre_rope = true;
    calib_metric_t metric = METRIC_VAR;
    float tq_threshold = 101.0f; // default: no override
    ggml_type tq_override_type = GGML_TYPE_Q8_0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i+1 < argc) model_path = argv[++i];
        else if ((arg == "-f" || arg == "--file") && i+1 < argc) calib_file = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i+1 < argc) output_path = argv[++i];
        else if ((arg == "-n" || arg == "--n-tokens") && i+1 < argc) n_tokens_max = std::atoi(argv[++i]);
        else if ((arg == "-c" || arg == "--ctx-size") && i+1 < argc) n_ctx = std::atoi(argv[++i]);
        else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i+1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (arg == "--pre-rope") pre_rope = true;
        else if (arg == "--post-rope") pre_rope = false;
        else if (arg == "--stats" && i+1 < argc) stats_path = argv[++i];
        else if (arg == "--threshold" && i+1 < argc) tq_threshold = std::atof(argv[++i]);
        else if (arg == "--override-type" && i+1 < argc) {
            std::string t = argv[++i];
            if (t == "f16")       tq_override_type = GGML_TYPE_F16;
            else if (t == "q8_0") tq_override_type = GGML_TYPE_Q8_0;
            else if (t == "q4_0") tq_override_type = GGML_TYPE_Q4_0;
            else { fprintf(stderr, "error: unknown override type '%s'\n", t.c_str()); return 1; }
        }
        else if (arg == "--metric" && i+1 < argc) {
            std::string m = argv[++i];
            if (m == "mag") metric = METRIC_MAG;
            else if (m == "var") metric = METRIC_VAR;
            else if (m == "both") metric = METRIC_BOTH;
            else { fprintf(stderr, "error: unknown metric '%s'\n", m.c_str()); return 1; }
        }
        else { print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || calib_file.empty()) { print_usage(argv[0]); return 1; }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer    = llama_model_n_layer(model);
    const int n_q_heads  = llama_model_n_head(model);

    // Detect per-layer head_dim and n_kv_heads from GGUF metadata
    int head_dim_global = 0, head_dim_swa = 0;
    int n_kv_heads_global = 0;
    {
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (arch[0]) {
            char key[256], val[64];
            snprintf(key, sizeof(key), "%s.attention.key_length", arch);
            if (llama_model_meta_val_str(model, key, val, sizeof(val)) > 0)
                head_dim_global = atoi(val);
            snprintf(key, sizeof(key), "%s.attention.key_length_swa", arch);
            if (llama_model_meta_val_str(model, key, val, sizeof(val)) > 0)
                head_dim_swa = atoi(val);
        }
        if (head_dim_global <= 0)
            head_dim_global = llama_model_n_embd(model) / n_q_heads;
        if (head_dim_swa <= 0)
            head_dim_swa = head_dim_global;
        n_kv_heads_global = llama_model_n_head_kv(model);
    }

    // Build per-layer head_dim and n_kv_heads from metadata
    std::vector<int> kv_heads_per_layer(n_layer, n_kv_heads_global);
    std::vector<int> head_dim_per_layer(n_layer, head_dim_global);
    {
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (arch[0]) {
            // Parse head_count_kv array: [8, 8, 8, 8, 8, 2, ...]
            char key[256], val[2048];
            snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
            if (llama_model_meta_val_str(model, key, val, sizeof(val)) > 0) {
                std::vector<int> hkv;
                const char * p = val;
                while (*p) {
                    while (*p && !isdigit(*p)) p++;
                    if (*p) { hkv.push_back(atoi(p)); while (*p && isdigit(*p)) p++; }
                }
                // SWA layers have the majority (higher) kv head count
                int max_hkv = *std::max_element(hkv.begin(), hkv.end());
                for (int il = 0; il < n_layer && il < (int)hkv.size(); il++) {
                    kv_heads_per_layer[il] = hkv[il];
                    head_dim_per_layer[il] = (hkv[il] == max_hkv) ? head_dim_swa : head_dim_global;
                }
            }
        }
    }

    fprintf(stderr, "model: %d layers, head_dim=%d (global), head_dim_swa=%d\n",
            n_layer, head_dim_global, head_dim_swa);
    for (int il = 0; il < n_layer; il++) {
        if (head_dim_per_layer[il] != head_dim_global || kv_heads_per_layer[il] != n_kv_heads_global) {
            fprintf(stderr, "  layer %2d: head_dim=%d, n_kv_heads=%d%s\n",
                    il, head_dim_per_layer[il], kv_heads_per_layer[il],
                    head_dim_per_layer[il] == head_dim_swa ? " (SWA)" : " (global)");
        }
    }

    const char * metric_str = metric == METRIC_MAG ? "mag" : metric == METRIC_VAR ? "var" : "both";
    g_calib.init(n_layer, kv_heads_per_layer, head_dim_per_layer, pre_rope, metric);

    fprintf(stderr, "calibration: %d layers, %s, metric=%s, max %d tokens\n",
            g_calib.n_layers, pre_rope ? "pre-RoPE" : "post-RoPE", metric_str, n_tokens_max);

    auto tokens = tokenize_file(vocab, calib_file, n_tokens_max);
    if (tokens.empty()) { llama_model_free(model); return 1; }
    fprintf(stderr, "tokenized %zu tokens\n", tokens.size());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx; cparams.n_batch = n_ctx; cparams.n_ubatch = n_ctx;
    cparams.type_k = GGML_TYPE_F16; cparams.type_v = GGML_TYPE_F16;
    cparams.cb_eval = tq_collect_callback;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); llama_model_free(model); return 1; }

    int n_processed = 0;
    while (n_processed < (int)tokens.size()) {
        int n_batch = std::min(n_ctx, (int)tokens.size() - n_processed);
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int i = 0; i < n_batch; i++) {
            common_batch_add(batch, tokens[n_processed+i], n_processed+i, {0}, false);
        }
        llama_decode(ctx, batch);
        n_processed += n_batch;
        fprintf(stderr, "\r  processed %d / %zu tokens", n_processed, tokens.size());
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    fprintf(stderr, "\n");

    // Compact layer map (skip layers with no data — hybrid models)
    int n_kv_layers = 0;
    std::vector<int32_t> final_layer_map(n_layer, -1);
    std::vector<int> kv_cidx_map;
    for (int il = 0; il < n_layer; il++) {
        int cidx = g_calib.layer_map[il];
        if (cidx >= 0 && g_calib.counts[cidx] > 0) {
            final_layer_map[il] = n_kv_layers;
            kv_cidx_map.push_back(cidx);
            n_kv_layers++;
        }
    }
    if (n_kv_layers < g_calib.n_layers) {
        fprintf(stderr, "hybrid model: %d/%d layers have KV attention\n", n_kv_layers, g_calib.n_layers);
    }

    fprintf(stderr, "computing permutations...\n");

    // Per-layer permutations stored as variable-length blocks
    // perm_offsets[l] = offset into all_perms for layer l
    std::vector<size_t> perm_offsets(n_kv_layers);
    std::vector<uint8_t> all_perms;

    for (int l = 0; l < n_kv_layers; l++) {
        int old_cidx = kv_cidx_map[l];
        int n_heads  = g_calib.layer_n_kv_heads[old_cidx];
        int hdim     = g_calib.layer_head_dim[old_cidx];

        perm_offsets[l] = all_perms.size();
        all_perms.resize(all_perms.size() + (size_t)n_heads * hdim);

        for (int h = 0; h < n_heads; h++) {
            g_calib.compute_perm(old_cidx, h, all_perms.data() + perm_offsets[l] + (size_t)h * hdim);
        }
        fprintf(stderr, "  layer %2d: %lld vectors, head_dim=%d, n_kv_heads=%d\n",
                l, (long long)g_calib.counts[old_cidx], hdim, n_heads);
    }

    // Compute outlier concentration per layer (top 25% channels' share of total variance)
    std::vector<float> outlier_pcts(n_kv_layers);
    std::vector<int32_t> layer_types(n_kv_layers);
    for (int l = 0; l < n_kv_layers; l++) {
        int old_cidx = kv_cidx_map[l];
        int n_heads  = g_calib.layer_n_kv_heads[old_cidx];
        int hdim     = g_calib.layer_head_dim[old_cidx];
        int n_hi     = hdim / 4;
        int64_t n = std::max(g_calib.counts[old_cidx], (int64_t)1);
        double total_var = 0, hi_var = 0;
        for (int h = 0; h < n_heads; h++) {
            size_t off = ((size_t)old_cidx * g_calib.max_n_kv_heads + h) * g_calib.max_head_dim;
            const double * sv = g_calib.sum_val.data() + off;
            const double * ss = g_calib.sum_sq.data()  + off;
            const uint8_t * perm = all_perms.data() + perm_offsets[l] + (size_t)h * hdim;
            for (int d = 0; d < hdim; d++) {
                double var = ss[d] / n - (sv[d] / n) * (sv[d] / n);
                if (var < 0) var = 0;
                total_var += var;
            }
            for (int d = 0; d < n_hi; d++) {
                int ch = perm[d];
                double var = ss[ch] / n - (sv[ch] / n) * (sv[ch] / n);
                if (var < 0) var = 0;
                hi_var += var;
            }
        }
        float pct = (float)(100.0 * hi_var / (total_var + 1e-30));
        outlier_pcts[l] = pct;
        if (pct > tq_threshold) {
            layer_types[l] = tq_override_type;
        } else {
            layer_types[l] = 0;
        }
        const char * rec = layer_types[l] ? ggml_type_name((ggml_type)layer_types[l]) : "tq";
        fprintf(stderr, "  layer %2d: top %d/%d channels hold %.1f%% of total variance -> %s\n",
                l, n_hi, hdim, pct, rec);
    }

    // Optional stats CSV
    if (!stats_path.empty()) {
        FILE * sf = fopen(stats_path.c_str(), "w");
        if (sf) {
            fprintf(sf, "layer,head,head_dim,channel,mean_abs,variance,std,rank\n");
            for (int l = 0; l < n_kv_layers; l++) {
                int old_cidx = kv_cidx_map[l];
                int n_heads  = g_calib.layer_n_kv_heads[old_cidx];
                int hdim     = g_calib.layer_head_dim[old_cidx];
                int64_t n = std::max(g_calib.counts[old_cidx], (int64_t)1);
                for (int h = 0; h < n_heads; h++) {
                    const uint8_t * perm = all_perms.data() + perm_offsets[l] + (size_t)h * hdim;
                    size_t off = ((size_t)old_cidx * g_calib.max_n_kv_heads + h) * g_calib.max_head_dim;
                    const double * sa = g_calib.sum_abs.data() + off;
                    const double * sv = g_calib.sum_val.data() + off;
                    const double * ss = g_calib.sum_sq.data()  + off;

                    std::vector<int> rank_of(hdim);
                    for (int r = 0; r < hdim; r++) rank_of[perm[r]] = r;

                    for (int d = 0; d < hdim; d++) {
                        double mag = sa[d] / n;
                        double var = ss[d] / n - (sv[d] / n) * (sv[d] / n);
                        if (var < 0) var = 0;
                        fprintf(sf, "%d,%d,%d,%d,%.8f,%.8f,%.8f,%d\n",
                                l, h, hdim, d, mag, var, sqrt(var), rank_of[d]);
                    }
                }
            }
            fclose(sf);
            fprintf(stderr, "saved channel statistics to '%s'\n", stats_path.c_str());
        }
    }

    // Write permutation file (v3: per-layer variable dimensions)
    FILE * fp = fopen(output_path.c_str(), "wb");
    if (!fp) { fprintf(stderr, "error: failed to open '%s'\n", output_path.c_str()); return 1; }
    uint32_t magic = 0x54515045, version = 3;
    uint32_t nl = n_kv_layers, nlm = n_layer;
    uint32_t pr = pre_rope ? 1 : 0;
    fwrite(&magic, 4, 1, fp);
    fwrite(&version, 4, 1, fp);
    fwrite(&nl, 4, 1, fp);
    fwrite(&pr, 4, 1, fp);
    fwrite(&nlm, 4, 1, fp);

    // Model layer → KV layer map
    for (int il = 0; il < n_layer; il++) {
        int32_t idx = final_layer_map[il];
        fwrite(&idx, 4, 1, fp);
    }

    // Per-layer header: [n_kv_heads, head_dim] for each KV layer
    for (int l = 0; l < n_kv_layers; l++) {
        int old_cidx = kv_cidx_map[l];
        uint32_t nh = g_calib.layer_n_kv_heads[old_cidx];
        uint32_t hd = g_calib.layer_head_dim[old_cidx];
        fwrite(&nh, 4, 1, fp);
        fwrite(&hd, 4, 1, fp);
    }

    // Permutation data (variable length per layer)
    fwrite(all_perms.data(), 1, all_perms.size(), fp);

    // TQLT section: per-layer type recommendations
    {
        uint32_t tqlt_magic = 0x54514C54;
        uint32_t n_entries = n_kv_layers;
        fwrite(&tqlt_magic, 4, 1, fp);
        fwrite(&n_entries, 4, 1, fp);
        fwrite(layer_types.data(), sizeof(int32_t), n_kv_layers, fp);
        fwrite(outlier_pcts.data(), sizeof(float), n_kv_layers, fp);
    }
    fclose(fp);

    fprintf(stderr, "saved to '%s' (%d KV layers, %s)\n",
            output_path.c_str(), n_kv_layers, pre_rope ? "pre-RoPE" : "post-RoPE");
    if (n_kv_layers > 0) {
        int cidx0 = kv_cidx_map[0];
        int hdim0 = g_calib.layer_head_dim[cidx0];
        fprintf(stderr, "sample layer 0, head 0 (top 16): ");
        for (int i = 0; i < std::min(16, hdim0); i++) fprintf(stderr, "%d ", all_perms[i]);
        fprintf(stderr, "...\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
