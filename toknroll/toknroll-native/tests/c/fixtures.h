/* Two tiny tokenizers the tests write themselves, so every id is known and nothing is downloaded: the
 * same byte-level BPE as a tokenizer.json and as a GGUF.
 *
 *   ids 0..255   byte b is id b, spelled as GPT-2's byte-level alphabet spells it ("Ġ" for a space)
 *   256          "he", the one merge (h e)
 *   257          "<|end|>", a special (control) token
 *   258          "<user>", an added token that is not special (user-defined)
 *   259..261     the GGUF only: "<unk>" (unknown), "<unused>" (unused), "<0x41>" (byte)
 *
 * "hello" is [256, 108, 108, 111]. write_gapped_json moves "he" to 300, so 256 and 259..299 are ids
 * no token has, a gap such as a real vocabulary can leave. */
#ifndef FIXTURES_H
#define FIXTURES_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  include <process.h>
#  include <windows.h>
#else
#  include <unistd.h>
#endif

/* A fixture appears whole or not at all: written under a name of this process's own, then renamed into
 * place, so scenarios running at once that write the same fixture never read one half-written. */
static inline FILE* fixture_open(const char* path, char* temp, size_t size) {
#ifdef _WIN32
    snprintf(temp, size, "%s.%d", path, (int) _getpid());
#else
    snprintf(temp, size, "%s.%d", path, (int) getpid());
#endif
    FILE* f = fopen(temp, "wb");
    if (!f) { printf("    cannot write %s\n", temp); exit(2); }
    return f;
}
static inline void fixture_close(FILE* f, const char* temp, const char* path) {
#ifdef _WIN32
    int ok = fclose(f) == 0 && MoveFileExA(temp, path, MOVEFILE_REPLACE_EXISTING);
#else
    int ok = fclose(f) == 0 && rename(temp, path) == 0;
#endif
    if (!ok) { printf("    cannot write %s\n", path); exit(2); }
}

enum { ID_HE = 256, ID_END = 257, ID_USER = 258, ID_UNK = 259, ID_UNUSED = 260, ID_BYTE = 261 };
enum { VOCAB_JSON = 259, VOCAB_GGUF = 262 };

/* GPT-2's bytes_to_unicode: the printable bytes stand for themselves, the rest for 256 + n */
static inline void byte_spellings(char spelling[256][4]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        int cp = printable ? b : 256 + n++;
        char* s = spelling[b];
        if (cp < 0x80) { s[0] = (char) cp; s[1] = 0; }
        else { s[0] = (char) (0xC0 | (cp >> 6)); s[1] = (char) (0x80 | (cp & 0x3F)); s[2] = 0; }
    }
}

static inline void write_tiny_json_he(const char* path, int he) {
    char spelling[256][4];
    byte_spellings(spelling);
    char temp[1024];
    FILE* f = fixture_open(path, temp, sizeof temp);
    fputs("{\"version\":\"1.0\",\"added_tokens\":["
          "{\"id\":257,\"content\":\"<|end|>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
          "{\"id\":258,\"content\":\"<user>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false}],"
          "\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true},"
          "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},"
          "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,"
          "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{", f);
    for (int b = 0; b < 256; b++) {
        const char* s = spelling[b];
        fprintf(f, "\"%s%s\":%d,", (*s == '"' || *s == '\\') ? "\\" : "", s, b);
    }
    fprintf(f, "\"he\":%d},\"merges\":[\"h e\"]}}", he);
    fixture_close(f, temp, path);
}
static inline void write_tiny_json(const char* path) { write_tiny_json_he(path, ID_HE); }
static inline void write_gapped_json(const char* path) { write_tiny_json_he(path, 300); }

static inline void gguf_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }   /* GGUF is little-endian, as every target is */
static inline void gguf_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
static inline void gguf_str(FILE* f, const char* s) { gguf_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }

/* SentencePiece's way, as Llama's tokenizer.json writes it: a word's space is a leading U+2581, and
 * the decoder drops a text's first one, so "hi there" is ids {7, 12} and 12 alone decodes to "there" */
static inline void write_metaspace_json(const char* path) {
    char temp[1024];
    FILE* f = fixture_open(path, temp, sizeof temp);
    fputs("{\"version\":\"1.0\",\"added_tokens\":[],\"normalizer\":null,\"post_processor\":null,"
          "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"\\u2581\",\"prepend_scheme\":\"always\",\"split\":true},"
          "\"decoder\":{\"type\":\"Metaspace\",\"replacement\":\"\\u2581\",\"prepend_scheme\":\"always\",\"split\":true},"
          "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,"
          "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{\"\\u2581\":0,\"h\":1,\"i\":2,\"t\":3,"
          "\"e\":4,\"r\":5,\"\\u2581h\":6,\"\\u2581hi\":7,\"\\u2581t\":8,\"\\u2581th\":9,\"\\u2581the\":10,\"\\u2581ther\":11,"
          "\"\\u2581there\":12,\"th\":13},\"merges\":[\"\\u2581 h\",\"\\u2581h i\",\"\\u2581 t\",\"t h\",\"\\u2581t h\",\"\\u2581th e\","
          "\"\\u2581the r\",\"\\u2581ther e\"]}}", f);
    fixture_close(f, temp, path);
}

/* tokenizer.ggml.pre is pre: "gpt2", or a name only an option can make known */
static inline void write_tiny_gguf(const char* path, const char* pre) {
    static const char* extra[] = {"he", "<|end|>", "<user>", "<unk>", "<unused>", "<0x41>"};
    static const int32_t extra_type[] = {1, 3, 4, 2, 5, 6};   /* normal, control, user-defined, unknown, unused, byte */
    char spelling[256][4];
    byte_spellings(spelling);
    char temp[1024];
    FILE* f = fixture_open(path, temp, sizeof temp);
    fwrite("GGUF", 1, 4, f);
    gguf_u32(f, 3);   /* version */
    gguf_u64(f, 0);   /* tensors */
    gguf_u64(f, 5);   /* key-value pairs */
    gguf_str(f, "tokenizer.ggml.model"); gguf_u32(f, 8); gguf_str(f, "gpt2");
    gguf_str(f, "tokenizer.ggml.pre"); gguf_u32(f, 8); gguf_str(f, pre);
    gguf_str(f, "tokenizer.ggml.tokens"); gguf_u32(f, 9); gguf_u32(f, 8); gguf_u64(f, VOCAB_GGUF);
    for (int b = 0; b < 256; b++) gguf_str(f, spelling[b]);
    for (int i = 0; i < 6; i++) gguf_str(f, extra[i]);
    gguf_str(f, "tokenizer.ggml.token_type"); gguf_u32(f, 9); gguf_u32(f, 5); gguf_u64(f, VOCAB_GGUF);
    for (int b = 0; b < 256; b++) { int32_t normal = 1; fwrite(&normal, 4, 1, f); }
    fwrite(extra_type, 4, 6, f);
    gguf_str(f, "tokenizer.ggml.merges"); gguf_u32(f, 9); gguf_u32(f, 8); gguf_u64(f, 1); gguf_str(f, "h e");
    fixture_close(f, temp, path);
}

#endif
