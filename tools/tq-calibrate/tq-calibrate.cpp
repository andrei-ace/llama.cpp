// TurboQuant offline calibration tool
// Accumulates per-head K channel magnitudes, computes per-layer-per-head
// channel permutations (outlier detection by |K| magnitude), and saves them.
//
// Usage: llama-tq-calibrate -m model.gguf -f calibration.txt -o perms.bin [options]

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include <functional>

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

enum calib_metric_t { METRIC_MAG = 0, METRIC_VAR = 1, METRIC_BOTH = 2 };

struct tq_calibration_data {
    int n_layers   = 0;
    int n_kv_heads = 0;
    int head_dim   = 0;
    bool pre_rope  = false;
    calib_metric_t metric = METRIC_VAR;

    // [n_layers][n_kv_heads][head_dim]
    std::vector<double> sum_abs;  // sum |K[d]|
    std::vector<double> sum_val;  // sum K[d]      (for variance: mean)
    std::vector<double> sum_sq;   // sum K[d]^2    (for variance: E[X^2])
    std::vector<int64_t> counts;
    std::vector<int> layer_map;
    std::mutex mtx;

    void init(int n_layer_model, int n_kv_heads_, int head_dim_, bool pre_rope_,
              calib_metric_t metric_, const std::function<bool(int)> & has_kv) {
        n_kv_heads = n_kv_heads_;
        head_dim   = head_dim_;
        pre_rope   = pre_rope_;
        metric     = metric_;

        layer_map.resize(n_layer_model, -1);
        int idx = 0;
        for (int il = 0; il < n_layer_model; il++) {
            if (has_kv(il)) layer_map[il] = idx++;
        }
        n_layers = idx;
        size_t sz = (size_t)n_layers * n_kv_heads * head_dim;
        sum_abs.resize(sz, 0.0);
        sum_val.resize(sz, 0.0);
        sum_sq.resize(sz, 0.0);
        counts.resize(n_layers, 0);
    }

    void accum_head(int cidx, int h, const float * hd) {
        size_t off = ((size_t)cidx * n_kv_heads + h) * head_dim;
        double * sa = sum_abs.data() + off;
        double * sv = sum_val.data() + off;
        double * ss = sum_sq.data()  + off;
        for (int d = 0; d < head_dim; d++) {
            double v = (double)hd[d];
            sa[d] += fabs(v);
            sv[d] += v;
            ss[d] += v * v;
        }
    }

    void accumulate(int model_il, const float * data, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;
        std::lock_guard<std::mutex> lock(mtx);
        for (int64_t t = 0; t < n_tokens; t++)
            for (int h = 0; h < n_kv_heads; h++)
                accum_head(cidx, h, data + (t * n_kv_heads + h) * head_dim);
        counts[cidx] += n_tokens;
    }

    void accumulate_2d(int model_il, const float * data, int64_t n_embd_k_gqa, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;
        std::lock_guard<std::mutex> lock(mtx);
        for (int64_t t = 0; t < n_tokens; t++)
            for (int h = 0; h < n_kv_heads; h++)
                accum_head(cidx, h, data + t * n_embd_k_gqa + h * head_dim);
        counts[cidx] += n_tokens;
    }

    void compute_perm(int cidx, int head, uint8_t * perm) const {
        size_t off = ((size_t)cidx * n_kv_heads + head) * head_dim;
        const double * sa = sum_abs.data() + off;
        const double * sv = sum_val.data() + off;
        const double * ss = sum_sq.data()  + off;
        int64_t n = std::max(counts[cidx], (int64_t)1);

        std::vector<std::pair<double, int>> importance(head_dim);
        for (int d = 0; d < head_dim; d++) {
            double mag = sa[d] / n;                          // mean |K|
            double var = ss[d] / n - (sv[d] / n) * (sv[d] / n); // Var(K)
            if (var < 0) var = 0; // numerical safety

            double score;
            if (metric == METRIC_MAG)       score = mag;
            else if (metric == METRIC_VAR)  score = var;
            else /* METRIC_BOTH */          score = mag * sqrt(var); // magnitude × std
            importance[d] = { score, d };
        }

        std::sort(importance.begin(), importance.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int d = 0; d < head_dim; d++) perm[d] = (uint8_t)importance[d].second;
    }
};

static tq_calibration_data g_calib;
static std::vector<char> g_tensor_buf;

static bool tq_collect_callback(struct ggml_tensor * t, bool ask, void *) {
    if (ask) {
        int il_tmp;
        if (sscanf(t->name, "Kcur-%d", &il_tmp) == 1) {
            return g_calib.pre_rope ? (ggml_n_dims(t) == 2) : (ggml_n_dims(t) == 3);
        }
        return false;
    }

    if (t->type != GGML_TYPE_F32) return true;
    int il = -1;
    if (sscanf(t->name, "Kcur-%d", &il) != 1) return true;
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

    if (g_calib.pre_rope) {
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
    fprintf(stderr, "       --pre-rope        Capture pre-RoPE K (default: post-RoPE)\n");
    fprintf(stderr, "       --post-rope       Capture post-RoPE K (default)\n");
    fprintf(stderr, "       --metric var      Outlier = highest variance Var(K) (default)\n");
    fprintf(stderr, "       --metric mag      Outlier = highest mean |K|\n");
    fprintf(stderr, "       --metric both     Outlier = |K| × std(K)\n");
}

int main(int argc, char ** argv) {
    std::string model_path, calib_file, output_path = "tq-perms.bin";
    int n_tokens_max = 4096, n_ctx = 512, n_gpu_layers = 99;
    bool pre_rope = false;
    calib_metric_t metric = METRIC_VAR;

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
    const int n_kv_heads = llama_model_n_head_kv(model);
    const int n_q_heads  = llama_model_n_head(model);
    const int head_dim   = llama_model_n_embd(model) / n_q_heads;

    fprintf(stderr, "model: %d layers, %d KV heads, %d Q heads (GQA %d:1), head_dim=%d\n",
            n_layer, n_kv_heads, n_q_heads, n_q_heads/n_kv_heads, head_dim);

    const char * metric_str = metric == METRIC_MAG ? "mag" : metric == METRIC_VAR ? "var" : "both";

    auto has_kv = [&](int) -> bool { return true; };
    g_calib.init(n_layer, n_kv_heads, head_dim, pre_rope, metric, has_kv);

    fprintf(stderr, "calibration: %d KV layers, %s, metric=%s, max %d tokens\n",
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
        for (int i = 0; i < n_batch; i++) common_batch_add(batch, tokens[n_processed+i], n_processed+i, {0}, false);
        llama_decode(ctx, batch);
        n_processed += n_batch;
        fprintf(stderr, "\r  processed %d / %zu tokens", n_processed, tokens.size());
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "computing permutations...\n");
    const int n_hi = head_dim / 4;
    std::vector<uint8_t> all_perms((size_t)g_calib.n_layers * n_kv_heads * head_dim);

    for (int l = 0; l < g_calib.n_layers; l++) {
        for (int h = 0; h < n_kv_heads; h++) {
            g_calib.compute_perm(l, h, all_perms.data() + ((size_t)l * n_kv_heads + h) * head_dim);
        }
        fprintf(stderr, "  layer %2d: %lld vectors\n", l, (long long)g_calib.counts[l]);
    }

    FILE * fp = fopen(output_path.c_str(), "wb");
    if (!fp) { fprintf(stderr, "error: failed to open '%s'\n", output_path.c_str()); return 1; }
    uint32_t magic = 0x54515045, version = 1;
    uint32_t nl = g_calib.n_layers, nh = n_kv_heads, hd = head_dim;
    uint32_t pr = pre_rope ? 1 : 0, nlm = n_layer;
    fwrite(&magic, 4, 1, fp); fwrite(&version, 4, 1, fp);
    fwrite(&nl, 4, 1, fp); fwrite(&nh, 4, 1, fp); fwrite(&hd, 4, 1, fp);
    fwrite(&pr, 4, 1, fp); fwrite(&nlm, 4, 1, fp);
    for (int il = 0; il < n_layer; il++) { int32_t idx = g_calib.layer_map[il]; fwrite(&idx, 4, 1, fp); }
    fwrite(all_perms.data(), 1, all_perms.size(), fp);
    fclose(fp);

    fprintf(stderr, "saved %zu bytes to '%s' (%d layers × %d heads × %d ch, %s)\n",
            28 + n_layer*4 + all_perms.size(), output_path.c_str(),
            g_calib.n_layers, n_kv_heads, head_dim, pre_rope ? "pre-RoPE" : "post-RoPE");
    fprintf(stderr, "outlier channels per head: %d (top %d%%)\n", n_hi, 100*n_hi/head_dim);
    fprintf(stderr, "sample layer 0, head 0: ");
    for (int i = 0; i < n_hi && i < 16; i++) fprintf(stderr, "%d ", all_perms[i]);
    if (n_hi > 16) fprintf(stderr, "...");
    fprintf(stderr, "\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
