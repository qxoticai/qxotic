// The LFM2 tokenizer component: GGUF-loaded vocabulary + special-token handling + the
// chat-template program, kept separate from the GGUF model/tensor loader (ModelLoader).
package com.qxotic.jinfer;

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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;

/** The universal GGUF tokenizer (toknroll-backed): vocabulary, special tokens, and the optional
 *  compiled chat template. Model-family pre-tokenizers (e.g. lfm2) are registered internally; the
 *  template compiler is injected because jinfer-core does not depend on jinfer-jinja. */
public class GgufTokenizer {

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
    private Specials specialsEncoder;
    private final CompiledTemplate chatTemplate;

    /** Tokenizer with the GGUF's chat template compiled through {@code templateCompiler}
     *  (typically {@code JinjaRenderer::template} — injected because jinfer-core doesn't depend
     *  on jinfer-jinja). */
    public GgufTokenizer(GGUF gguf, Function<String, CompiledTemplate> templateCompiler) {
        this.tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins()
                .registerPreTokenizer("lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                .registerNormalizer("lfm2", g -> Normalizer.identity())
                .build()
                .fromGGUF(gguf);
        String raw = gguf.getStringOrDefault("tokenizer.chat_template", "");
        this.chatTemplate = !raw.isEmpty() ? templateCompiler.apply(raw) : null;
        this.specialTokens = new HashMap<>();
        for (int id = 0; id < vocabularySize(); id++) {
            if (isSpecialToken(id)) {
                specialTokens.put(tokenizer.vocabulary().token(id), id);
            }
        }
    }

    public int vocabularySize() {
        return tokenizer.vocabulary().size();
    }

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    /** The compiled chat template, or null when the GGUF carries none (or it failed to
     *  compile) — chat requests then fail with a descriptive error; raw prompts still work. */
    public CompiledTemplate chatTemplate() { return chatTemplate; }

    boolean isSpecialToken(int token) {
        return token >= 0 && token < vocabularySize()
                && !tokenizer.vocabulary().isTokenOfType(token, StandardTokenType.NORMAL);
    }

    public List<Integer> encode(String text) {
        return tokenizer.encode(text).toList();
    }

    /** Encode mapping special-token strings in the text to their ids (plain {@link #encode}
     *  never maps them); the --raw-prompt path uses this to author templated streams as text. */
    public List<Integer> encodeWithSpecialTokens(String text) {
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
    private Set<String> specialMatchSet() {
        Set<String> keys = specialTokens.keySet();
        Set<String> kept = new HashSet<>(keys);
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

    public String decode(int token) {
        return new String(decodeTokenBytes(token), StandardCharsets.UTF_8);
    }

    public String decode(List<Integer> tokens) {
        var buf = new ByteArrayOutputStream();
        for (int token : tokens) {
            buf.writeBytes(tokenizer.decodeBytes(new int[]{token}));
        }
        return buf.toString(StandardCharsets.UTF_8);
    }
}
