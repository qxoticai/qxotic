// llama.cpp oracle for the LFM3 parity tests: dumps every named graph tensor of one forward.
//
//   text   <model.gguf> <prompt.txt> <out-dir>                 one batch over the whole prompt
//   media  <model.gguf> <mmproj.gguf> <media-file> <out-dir>   one mtmd encode (image or audio)
//   speech <model.gguf> <mmproj.gguf> <prompt.txt> <out-dir>   LFM3-Omni speech output, greedy
//
// Text runs with an F32 KV cache and no flash attention, so the reference keeps F32 scores; it
// writes tokens.i32 and logits.f32 [n_tokens][n_vocab]. Media writes embd.f32 [rows][n_embd].
// Every tensor the graph computes lands in <name>.t: int64 ne[4], then F32 data, ne0 fastest.
//
// Build (from this directory, llama.cpp checked out next to qxotic):
//   L=../../../llama.cpp; g++ -O2 -std=c++17 lfm3-oracle.cpp -o lfm3-oracle -I$L/include \
//     -I$L/ggml/include -I$L/tools/mtmd -L$L/build/bin -lllama -lmtmd -lggml -lggml-base \
//     -Wl,-rpath,$L/build/bin
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::string out_dir;
static std::map<std::string, int> seen;

static void write_file(const std::string & name, const void * data, size_t bytes) {
    FILE * f = fopen((out_dir + "/" + name).c_str(), "wb");
    if (!f) { perror(name.c_str()); exit(1); }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

static float element(const ggml_tensor * t, const uint8_t * data, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const uint8_t * p = data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    switch (t->type) {
        case GGML_TYPE_F32: return *(const float *) p;
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        case GGML_TYPE_I32: return (float) *(const int32_t *) p;
        default: return NAN;
    }
}

static bool dump_tensor(ggml_tensor * t, bool ask, void *);

// every float tensor once, under its graph name (a repeated name gets a #k suffix); the media
// input (inp_raw: pixels or mel) is a leaf, so it is written when its first consumer runs
static bool dump_tensor(ggml_tensor * t, bool ask, void * user) {
    if (ask) return true;
    for (ggml_tensor * src : t->src)
        if (src && strcmp(src->name, "inp_raw") == 0 && seen.count("inp_raw") == 0) dump_tensor(src, false, user);
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) return true;
    if (t->name[0] == '\0') return true;
    std::string name = t->name;
    // attention scores, cache views and anonymous views: huge, and the named tensors around them pin them
    // attention scores, cache views and derived views (permuted, reshaped, ...): huge, or copies
    if (name.rfind("kq-", 0) == 0 || name.rfind("kq_soft_max", 0) == 0 || name.rfind("cache_", 0) == 0 || name.find(" (") != std::string::npos)
        return true;
    for (char & c : name) if (c == '/' || c == ' ' || c == '(' || c == ')') c = '_';
    int k = seen[name]++;
    if (k > 0) name += "#" + std::to_string(k);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> out;
    out.reserve(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++)
                    out.push_back(element(t, raw.data(), i0, i1, i2, i3));
    std::vector<uint8_t> blob(4 * sizeof(int64_t) + out.size() * sizeof(float));
    memcpy(blob.data(), t->ne, 4 * sizeof(int64_t));
    memcpy(blob.data() + 4 * sizeof(int64_t), out.data(), out.size() * sizeof(float));
    write_file(name + ".t", blob.data(), blob.size());
    return true;
}

static int run_text(const char * model_path, const char * prompt_path) {
    std::ifstream in(prompt_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string prompt = buffer.str();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), tokens.data(), (int) tokens.size(),
                           llama_vocab_get_add_bos(vocab), true);
    if (n < 0) return 1;
    tokens.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = cp.n_ubatch = n;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = dump_tensor;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) return 1;

    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> logits((size_t) n * n_vocab);
    for (int i = 0; i < n; i++) memcpy(&logits[(size_t) i * n_vocab], llama_get_logits_ith(ctx, i), n_vocab * sizeof(float));
    write_file("tokens.i32", tokens.data(), tokens.size() * sizeof(llama_token));
    write_file("logits.f32", logits.data(), logits.size() * sizeof(float));
    fprintf(stderr, "text: %d tokens, %zu tensors\n", n, seen.size());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int run_media(const char * model_path, const char * mmproj_path, const char * media_path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;

    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = false;
    p.warmup = false;
    p.cb_eval = dump_tensor;
    mtmd_context * ctx = mtmd_init_from_file(mmproj_path, model, p);
    if (!ctx) return 1;
    seen.clear();

    mtmd::bitmap bitmap(mtmd_helper_bitmap_init_from_file(ctx, media_path, false, mtmd_helper_init_opt_default()).bitmap);
    if (!bitmap.ptr) return 1;
    std::string text = mtmd_get_marker(ctx);
    mtmd_input_text input = {text.c_str(), text.size(), false, true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    const mtmd_bitmap * bitmaps[] = {bitmap.ptr.get()};
    if (mtmd_tokenize(ctx, chunks.ptr.get(), &input, bitmaps, 1) != 0) return 1;
    for (size_t i = 0; i < chunks.size(); i++) {
        const mtmd_input_chunk * chunk = chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
        if (mtmd_encode_chunk(ctx, chunk) != 0) return 1;
        const size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
        const int n_embd = llama_model_n_embd(model);
        write_file("embd.f32", mtmd_get_output_embd(ctx), rows * n_embd * sizeof(float));
        fprintf(stderr, "media: %zu rows x %d, %zu tensors\n", rows, n_embd, seen.size());
    }
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}

// Greedy speech from a rendered TTS prompt, deterministic where the pipeline is random: codebooks
// revealed in index order (4 per depthformer pass) and argmax codes, no repetition penalty.
// Writes h.f32 (the backbone hidden state after the prompt), logits{0,1,2}.f32 [12][vocab] of the
// first frame's three passes, fb.f32 (its feedback row), and for every frame: codes.i32, hidden
// states hs.f32, and pcm.f32 (codec + ISTFT, the tail flushed).
static int run_speech(const char * model_path, const char * mmproj_path, const char * prompt_path, int n_frames) {
    std::ifstream in(prompt_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string prompt = buffer.str();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), tokens.data(), (int) tokens.size(), false, true);
    tokens.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + n_frames + 16;
    cp.n_batch = cp.n_ubatch = n;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings = true;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_embd = llama_model_n_embd(model);

    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = false;
    p.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, p);
    if (!mctx) return 1;
    mtmd::gen_state_ptr state(mtmd_gen_state_init(mctx, mtmd_gen_state_params_default()));

    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_decode(ctx, batch) != 0) return 1;
    write_file("tokens.i32", tokens.data(), tokens.size() * sizeof(llama_token));

    const int n_cb = 12;
    std::vector<int32_t> all_codes;
    std::vector<float> hs, pcm;
    int pos = n;
    for (int frame = 0; frame < n_frames; frame++) {
        std::vector<float> h(llama_get_embeddings_ith(ctx, -1), llama_get_embeddings_ith(ctx, -1) + n_embd);
        hs.insert(hs.end(), h.begin(), h.end());
        if (frame == 0) write_file("h.f32", h.data(), h.size() * sizeof(float));
        std::vector<int32_t> codes(n_cb, -1);
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd = h.data();
        inp.codes = codes.data();
        inp.n_codes = codes.size();
        mtmd_gen_out out{};
        bool eos = false;
        for (int pass = 0; pass < 3; pass++) {
            if (mtmd_gen_audio_process(mctx, state.get(), &inp, &out) != 0) return 1;
            const size_t n_vocab = out.n_feats / n_cb;
            if (frame == 0) write_file("logits" + std::to_string(pass) + ".f32", out.feats, out.n_feats * sizeof(float));
            for (int i = pass * 4; i < pass * 4 + 4; i++) {
                const float * l = out.feats + (size_t) i * n_vocab;
                codes[i] = (int32_t) (std::max_element(l, l + n_vocab) - l);
                if (codes[i] == (int32_t) n_vocab - 1) eos = true;
            }
        }
        if (eos) {
            fprintf(stderr, "speech: end of audio at frame %d\n", frame);
            break;
        }
        if (mtmd_gen_audio_process(mctx, state.get(), &inp, &out) != 0) return 1;
        if (frame == 0) write_file("fb.f32", out.embd, n_embd * sizeof(float));
        std::vector<float> fb(out.embd, out.embd + n_embd);
        all_codes.insert(all_codes.end(), codes.begin(), codes.end());

        mtmd_gen_inp wav = mtmd_gen_inp_default(mctx);
        wav.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        wav.codes = codes.data();
        wav.n_codes = codes.size();
        if (mtmd_gen_audio_process(mctx, state.get(), &wav, &out) != 0) return 1;
        pcm.insert(pcm.end(), out.audio, out.audio + out.n_samples);

        llama_batch eb = llama_batch_init(1, n_embd, 1);
        memcpy(eb.embd, fb.data(), n_embd * sizeof(float));
        eb.pos[0] = pos++;
        eb.n_seq_id[0] = 1;
        eb.seq_id[0][0] = 0;
        eb.logits[0] = true;
        eb.n_tokens = 1;
        if (llama_decode(ctx, eb) != 0) return 1;
        llama_batch_free(eb);
    }
    mtmd_gen_inp flush = mtmd_gen_inp_default(mctx);
    flush.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
    mtmd_gen_out out{};
    if (mtmd_gen_audio_process(mctx, state.get(), &flush, &out) != 0) return 1;
    pcm.insert(pcm.end(), out.audio, out.audio + out.n_samples);
    write_file("codes.i32", all_codes.data(), all_codes.size() * sizeof(int32_t));
    write_file("hs.f32", hs.data(), hs.size() * sizeof(float));
    write_file("pcm.f32", pcm.data(), pcm.size() * sizeof(float));
    fprintf(stderr, "speech: %zu frames, %zu samples\n", all_codes.size() / n_cb, pcm.size());
    return 0;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    if (argc == 5 && strcmp(argv[1], "text") == 0) {
        out_dir = argv[4];
        return run_text(argv[2], argv[3]);
    }
    if (argc == 6 && strcmp(argv[1], "media") == 0) {
        out_dir = argv[5];
        return run_media(argv[2], argv[3], argv[4]);
    }
    if (argc == 6 && strcmp(argv[1], "speech") == 0) {
        out_dir = argv[5];
        return run_speech(argv[2], argv[3], argv[4], 24);
    }
    fprintf(stderr, "usage: %s text <model> <prompt.txt> <out> | media|speech <model> <mmproj> <file> <out>\n", argv[0]);
    return 2;
}
