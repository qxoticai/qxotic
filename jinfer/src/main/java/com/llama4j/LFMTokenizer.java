// The LFM2 tokenizer component: GGUF-loaded vocabulary + special-token handling + the
// chat-template program, kept separate from the GGUF model/tensor loader (ModelLoader).
package com.llama4j;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * LFM2 tokenizer = com.qxotic:toknroll loaded from the GGUF metadata, plus the "lfm2"
 * pre-tokenizer registration (llama.cpp's regex; [\p{L}\p{M}]+ because Java's \p{L} misses
 * combining marks; no UNICODE_CHARACTER_CLASS so \s stays ASCII — token-identical to
 * llama-tokenize, see TokenizerParityTest). encode never maps special-token strings;
 * the chat format inserts special ids explicitly via {@link #getSpecialTokens()}.
 */
class LFMTokenizer {

    private static final String LFM2_PRE_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+" +
                    "|\\p{N}{1,3}" +
                    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                    "|\\s*[\\r\\n]+" +
                    "|\\s+(?!\\S)" +
                    "|\\s+";

    private final Tokenizer tokenizer;
    private final Map<String, Integer> specialTokens;
    private Specials specialsEncoder; // lazy: only the --raw-prompt path needs it
    private final JinjaRenderer.Prog chatTemplateProg; // compiled tokenizer.chat_template, or null

    LFMTokenizer(GGUF gguf) {
        this.tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins()
                .registerPreTokenizer("lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                .registerNormalizer("lfm2", g -> Normalizer.identity())
                .build()
                .fromGGUF(gguf);
        this.chatTemplateProg = compileChatTemplate(gguf.getStringOrDefault("tokenizer.chat_template", ""));
        this.specialTokens = new HashMap<>();
        for (int id = 0; id < vocabularySize(); id++) {
            if (isSpecialToken(id)) {
                specialTokens.put(tokenizer.vocabulary().token(id), id);
            }
        }
    }

    int vocabularySize() {
        return tokenizer.vocabulary().size();
    }

    Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    /** The compiled Jinja chat template, or null when the model has none or it failed to
     *  compile (the server then falls back to the built-in ChatML format). */
    JinjaRenderer.Prog chatTemplateProg() { return chatTemplateProg; }

    /** Compiles the chat template once at load. Returns null for an absent or malformed
     *  template so a single bad template degrades to ChatML instead of failing every request. */
    private static JinjaRenderer.Prog compileChatTemplate(String template) {
        if (template.isEmpty()) return null;
        try {
            return JinjaRenderer.compile(template);
        } catch (RuntimeException e) {
            System.err.println("[warn] chat template compilation failed: " + e.getMessage());
            return null;
        }
    }

    boolean isSpecialToken(int token) {
        return token >= 0 && token < vocabularySize()
                && !tokenizer.vocabulary().isTokenOfType(token, StandardTokenType.NORMAL);
    }

    List<Integer> encode(String text) {
        return tokenizer.encode(text).toList();
    }

    /** Encode mapping special-token strings in the text to their ids (plain {@link #encode}
     *  never maps them); the --raw-prompt path uses this to author templated streams as text. */
    List<Integer> encodeWithSpecialTokens(String text) {
        if (specialsEncoder == null) {
            specialsEncoder = Specials.compile(tokenizer.vocabulary(), specialMatchSet());
        }
        return specialsEncoder.encode(tokenizer, text).toList();
    }

    /** toknroll's special-token matcher rejects a set where one token is a strict prefix of another
     *  (e.g. MiniCPM's {@code <param} vs {@code <parameters>}). Drop the shorter token, which under
     *  longest-match is only reachable as a substring of the longer one — rendered control-token
     *  streams never emit it standalone, so this is loss-free in practice. No-op for models with no
     *  prefix-conflicting specials (all others). */
    private java.util.Set<String> specialMatchSet() {
        java.util.Set<String> keys = specialTokens.keySet();
        java.util.Set<String> kept = new java.util.HashSet<>(keys);
        for (String s : keys) {
            for (String o : keys) {
                if (!s.equals(o) && o.startsWith(s)) { kept.remove(s); break; }
            }
        }
        return kept;
    }

    /** Raw UTF-8 bytes of one token (the streaming decoder assembles code points across tokens). */
    byte[] decodeTokenBytes(int token) {
        return tokenizer.decodeBytes(new int[]{token});
    }

    String decode(int token) {
        return new String(decodeTokenBytes(token), StandardCharsets.UTF_8);
    }

    String decode(List<Integer> tokens) {
        var buf = new ByteArrayOutputStream();
        for (int token : tokens) {
            buf.writeBytes(tokenizer.decodeBytes(new int[]{token}));
        }
        return buf.toString(StandardCharsets.UTF_8);
    }
}
