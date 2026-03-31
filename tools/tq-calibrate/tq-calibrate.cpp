// TurboQuant offline calibration tool
// Three calibration modes: |K| magnitude, |Q·K| attention-weighted, or both combined.
// Supports pre-RoPE and post-RoPE capture.
//
// Usage: llama-tq-calibrate -m model.gguf -f calibration.txt -o perms.bin [options]

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

enum calib_mode_t {
    CALIB_K_MAG  = 0, // |K| per channel (original)
    CALIB_QK_DOT = 1, // |Q·K| per channel (attention-weighted)
    CALIB_BOTH   = 2, // geometric mean of both
};

struct tq_calibration_data {
    int n_layers    = 0;
    int n_kv_heads  = 0;
    int n_q_heads   = 0;
    int head_dim    = 0;
    int gqa_ratio   = 1;
    bool pre_rope   = false;
    calib_mode_t mode = CALIB_K_MAG;

    // [n_layers][n_kv_heads][head_dim]
    std::vector<double> accum_k;   // accumulated |K| per channel
    std::vector<double> accum_qk;  // accumulated |Q·K| per channel (GQA-averaged)
    std::vector<int64_t> counts_k;
    std::vector<int64_t> counts_qk;

    // K buffer: store last K seen per layer for Q·K computation
    // [n_layers][n_kv_heads * head_dim * max_tokens]
    std::vector<std::vector<float>> k_buf;  // per-layer K buffer
    std::vector<int64_t> k_buf_ntok;        // tokens in each K buffer

    std::vector<int> layer_map;
    std::mutex mtx;

    void init(int n_layer_model, int n_kv_heads_, int n_q_heads_, int head_dim_, bool pre_rope_,
              calib_mode_t mode_, const std::function<bool(int)> & has_kv) {
        n_kv_heads = n_kv_heads_;
        n_q_heads  = n_q_heads_;
        head_dim   = head_dim_;
        gqa_ratio  = n_q_heads / n_kv_heads;
        pre_rope   = pre_rope_;
        mode       = mode_;

        layer_map.resize(n_layer_model, -1);
        int idx = 0;
        for (int il = 0; il < n_layer_model; il++) {
            if (has_kv(il)) layer_map[il] = idx++;
        }
        n_layers = idx;

        size_t sz = (size_t)n_layers * n_kv_heads * head_dim;
        accum_k.resize(sz, 0.0);
        counts_k.resize(n_layers, 0);

        if (mode != CALIB_K_MAG) {
            accum_qk.resize(sz, 0.0);
            counts_qk.resize(n_layers, 0);
            k_buf.resize(n_layers);
            k_buf_ntok.resize(n_layers, 0);
        }
    }

    double * get_k(int cidx, int h)  { return accum_k.data()  + ((size_t)cidx * n_kv_heads + h) * head_dim; }
    double * get_qk(int cidx, int h) { return accum_qk.data() + ((size_t)cidx * n_kv_heads + h) * head_dim; }

    void accumulate_k(int model_il, const float * data, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0) return;
        std::lock_guard<std::mutex> lock(mtx);

        // Accumulate |K| per channel
        for (int64_t t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_kv_heads; h++) {
                double * acc = get_k(cidx, h);
                const float * hd = data + (t * n_kv_heads + h) * head_dim;
                for (int d = 0; d < head_dim; d++) acc[d] += fabs((double)hd[d]);
            }
        }
        counts_k[cidx] += n_tokens;

        // Buffer K for Q·K computation
        if (mode != CALIB_K_MAG) {
            size_t ksz = (size_t)n_kv_heads * head_dim * n_tokens;
            k_buf[cidx].resize(ksz);
            memcpy(k_buf[cidx].data(), data, ksz * sizeof(float));
            k_buf_ntok[cidx] = n_tokens;
        }
    }

    void accumulate_qk(int model_il, const float * q_data, int64_t n_tokens) {
        int cidx = layer_map[model_il];
        if (cidx < 0 || mode == CALIB_K_MAG) return;
        if (k_buf_ntok[cidx] == 0) return; // no K buffered yet

        std::lock_guard<std::mutex> lock(mtx);

        int64_t k_ntok = k_buf_ntok[cidx];
        int64_t ntok = std::min(n_tokens, k_ntok);
        const float * k_data = k_buf[cidx].data();

        // For each token, compute per-channel |Q_h · K| averaged across GQA Q heads
        for (int64_t t = 0; t < ntok; t++) {
            for (int kh = 0; kh < n_kv_heads; kh++) {
                double * acc = get_qk(cidx, kh);
                const float * k_head = k_data + (t * n_kv_heads + kh) * head_dim;

                // Average |Q_h[d] * K[d]| across the GQA Q heads sharing this KV head
                for (int d = 0; d < head_dim; d++) {
                    double sum = 0.0;
                    for (int g = 0; g < gqa_ratio; g++) {
                        int qh = kh * gqa_ratio + g;
                        const float * q_head = q_data + (t * n_q_heads + qh) * head_dim;
                        sum += fabs((double)q_head[d] * (double)k_head[d]);
                    }
                    acc[d] += sum / gqa_ratio;
                }
            }
        }
        counts_qk[cidx] += ntok;
    }

    void compute_perm(int cidx, int head, uint8_t * perm) const {
        std::vector<std::pair<double, int>> mags(head_dim);

        if (mode == CALIB_K_MAG) {
            const double * acc = accum_k.data() + ((size_t)cidx * n_kv_heads + head) * head_dim;
            int64_t cnt = std::max(counts_k[cidx], (int64_t)1);
            for (int d = 0; d < head_dim; d++) mags[d] = { acc[d] / cnt, d };
        } else if (mode == CALIB_QK_DOT) {
            const double * acc = accum_qk.data() + ((size_t)cidx * n_kv_heads + head) * head_dim;
            int64_t cnt = std::max(counts_qk[cidx], (int64_t)1);
            for (int d = 0; d < head_dim; d++) mags[d] = { acc[d] / cnt, d };
        } else { // CALIB_BOTH — geometric mean
            const double * ak = accum_k.data()  + ((size_t)cidx * n_kv_heads + head) * head_dim;
            const double * aq = accum_qk.data() + ((size_t)cidx * n_kv_heads + head) * head_dim;
            int64_t ck = std::max(counts_k[cidx], (int64_t)1);
            int64_t cq = std::max(counts_qk[cidx], (int64_t)1);
            for (int d = 0; d < head_dim; d++) {
                double mk = ak[d] / ck;
                double mq = aq[d] / cq;
                mags[d] = { sqrt(mk * mq), d };
            }
        }

        std::sort(mags.begin(), mags.end(), [](const auto & a, const auto & b) {
            return a.first > b.first;
        });
        for (int d = 0; d < head_dim; d++) perm[d] = (uint8_t)mags[d].second;
    }
};

static tq_calibration_data g_calib;
static std::vector<char> g_tensor_buf;
static std::vector<char> g_tensor_buf2;

static const float * read_tensor_data(struct ggml_tensor * t, std::vector<char> & buf) {
    const size_t nbytes = ggml_nbytes(t);
    if (ggml_backend_buffer_is_host(t->buffer)) return (const float *)t->data;
    if (buf.size() < nbytes) buf.resize(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    return (const float *)buf.data();
}

static bool tq_collect_callback(struct ggml_tensor * t, bool ask, void *) {
    if (ask) {
        int il_tmp;
        // Capture Kcur (3D post-RoPE or 2D pre-RoPE)
        if (sscanf(t->name, "Kcur-%d", &il_tmp) == 1) {
            if (g_calib.pre_rope) return (ggml_n_dims(t) == 2);
            else                  return (ggml_n_dims(t) == 3);
        }
        // Capture Qcur (3D post-RoPE) for Q·K mode
        if (g_calib.mode != CALIB_K_MAG) {
            if (sscanf(t->name, "Qcur-%d", &il_tmp) == 1) {
                return (ggml_n_dims(t) == 3); // [head_dim, n_q_heads, n_tokens]
            }
        }
        return false;
    }

    if (t->type != GGML_TYPE_F32) return true;

    int il = -1;
    bool is_k = (sscanf(t->name, "Kcur-%d", &il) == 1);
    bool is_q = false;
    if (!is_k) is_q = (sscanf(t->name, "Qcur-%d", &il) == 1);
    if (!is_k && !is_q) return true;
    if (il < 0 || il >= (int)g_calib.layer_map.size()) return true;
    if (g_calib.layer_map[il] < 0) return true;

    if (is_k) {
        const float * data = read_tensor_data(t, g_tensor_buf);
        if (g_calib.pre_rope) {
            // 2D pre-RoPE: reshape to 3D for uniform handling
            // [n_embd_k_gqa, n_tokens] → treat as [head_dim, n_kv_heads, n_tokens]
            g_calib.accumulate_k(il, data, t->ne[1]);
        } else {
            g_calib.accumulate_k(il, data, t->ne[2]);
        }
    } else if (is_q && g_calib.mode != CALIB_K_MAG) {
        // Qcur: [head_dim, n_q_heads, n_tokens]
        const float * data = read_tensor_data(t, g_tensor_buf2);
        g_calib.accumulate_qk(il, data, t->ne[2]);
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
    fprintf(stderr, "       --mode k          Channel importance = |K| magnitude (default)\n");
    fprintf(stderr, "       --mode qk         Channel importance = |Q·K| attention-weighted\n");
    fprintf(stderr, "       --mode both       Channel importance = geometric mean of |K| and |Q·K|\n");
}

int main(int argc, char ** argv) {
    std::string model_path, calib_file, output_path = "tq-perms.bin";
    int n_tokens_max = 4096, n_ctx = 512, n_gpu_layers = 99;
    bool pre_rope = false;
    calib_mode_t mode = CALIB_K_MAG;

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
        else if (arg == "--mode" && i+1 < argc) {
            std::string m = argv[++i];
            if (m == "k") mode = CALIB_K_MAG;
            else if (m == "qk") mode = CALIB_QK_DOT;
            else if (m == "both") mode = CALIB_BOTH;
            else { fprintf(stderr, "error: unknown mode '%s'\n", m.c_str()); return 1; }
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

    const char * mode_str = mode == CALIB_K_MAG ? "k" : mode == CALIB_QK_DOT ? "qk" : "both";
    fprintf(stderr, "model: %d layers, %d KV heads, %d Q heads (GQA %d:1), head_dim=%d\n",
            n_layer, n_kv_heads, n_q_heads, n_q_heads/n_kv_heads, head_dim);

    auto has_kv = [&](int) -> bool { return true; };
    g_calib.init(n_layer, n_kv_heads, n_q_heads, head_dim, pre_rope, mode, has_kv);

    fprintf(stderr, "calibration: %d KV layers, %s, mode=%s, max %d tokens\n",
            g_calib.n_layers, pre_rope ? "pre-RoPE" : "post-RoPE", mode_str, n_tokens_max);

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

    // Compute and save permutations
    fprintf(stderr, "computing permutations (mode=%s)...\n", mode_str);
    const int n_hi = head_dim / 4;
    std::vector<uint8_t> all_perms((size_t)g_calib.n_layers * n_kv_heads * head_dim);

    for (int l = 0; l < g_calib.n_layers; l++) {
        for (int h = 0; h < n_kv_heads; h++) {
            g_calib.compute_perm(l, h, all_perms.data() + ((size_t)l * n_kv_heads + h) * head_dim);
        }
        fprintf(stderr, "  layer %2d: K=%lld QK=%lld vectors\n", l,
                (long long)g_calib.counts_k[l],
                mode != CALIB_K_MAG ? (long long)g_calib.counts_qk[l] : 0LL);
    }

    // Save
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

    fprintf(stderr, "saved %zu bytes to '%s' (%d layers × %d heads × %d ch, mode=%s, %s)\n",
            28 + n_layer*4 + all_perms.size(), output_path.c_str(),
            g_calib.n_layers, n_kv_heads, head_dim, mode_str, pre_rope ? "pre-RoPE" : "post-RoPE");

    fprintf(stderr, "sample layer 0, head 0 top-%d outliers: ", n_hi);
    for (int i = 0; i < n_hi && i < 16; i++) fprintf(stderr, "%d ", all_perms[i]);
    if (n_hi > 16) fprintf(stderr, "...");
    fprintf(stderr, "\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
