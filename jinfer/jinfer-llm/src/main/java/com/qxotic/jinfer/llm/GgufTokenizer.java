// The LFM2 tokenizer component: GGUF-loaded vocabulary + special-token handling + the raw
// chat-template source, kept separate from the GGUF model/tensor loader (ModelLoader).
package com.qxotic.jinfer.llm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The universal GGUF tokenizer (toknroll-backed): vocabulary, special tokens, and the raw chat
 * template source. Model-family pre-tokenizers (e.g. lfm2) are registered internally. This layer
 * never compiles or renders the template — that is the Jinja engine's job, one layer up.
 */
public class GgufTokenizer {

    private static final String LFM2_PRE_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    private final Tokenizer tokenizer;
    private final Map<String, Integer> specialTokens;
    private Specials specialsEncoder;
    private final String chatTemplateSource;

    public GgufTokenizer(GGUF gguf) {
        this.tokenizer =
                GGUFTokenizerLoader.createBuilderWithBuiltins()
                        .registerPreTokenizer(
                                "lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                        .registerNormalizer("lfm2", g -> Normalizer.identity())
                        .build()
                        .fromGGUF(gguf);
        this.chatTemplateSource = gguf.getStringOrDefault("tokenizer.chat_template", "");
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

    /** The id of a special token a curated template requires; throws naming it when absent. */
    public int requiredSpecial(String name) {
        Integer id = specialTokens.get(name);
        if (id == null) throw new IllegalArgumentException("tokenizer lacks " + name);
        return id;
    }

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    /**
     * The GGUF's raw {@code tokenizer.chat_template} Jinja source, or {@code ""} when it carries
     * none. The tokenizer only transports the string; compiling and rendering it belongs to the
     * Jinja engine.
     */
    public String chatTemplateSource() {
        return chatTemplateSource;
    }

    public boolean isSpecialToken(int token) {
        return token >= 0
                && token < vocabularySize()
                && !tokenizer.vocabulary().isTokenOfType(token, StandardTokenType.NORMAL);
    }

    public IntSequence encode(String text) {
        return tokenizer.encode(text);
    }

    /**
     * Encode mapping special-token strings in the text to their ids (plain {@link #encode} never
     * maps them); the --raw-prompt path uses this to author templated streams as text.
     */
    public IntSequence encodeWithSpecialTokens(String text) {
        if (specialsEncoder == null) {
            specialsEncoder = Specials.compile(tokenizer.vocabulary(), specialMatchSet());
        }
        return specialsEncoder.encode(tokenizer, text);
    }

    /**
     * toknroll's special-token matcher rejects a set where one token is a strict prefix of another
     * (e.g. MiniCPM's {@code <param} vs {@code <parameters>}). Drop the shorter token, which under
     * longest-match is only reachable as a substring of the longer one — rendered control-token
     * streams never emit it standalone, so this is loss-free in practice. No-op for models with no
     * prefix-conflicting specials (all others).
     */
    private Set<String> specialMatchSet() {
        Set<String> keys = specialTokens.keySet();
        Set<String> kept = new HashSet<>(keys);
        for (String s : keys) {
            for (String o : keys) {
                if (!s.equals(o) && o.startsWith(s)) {
                    kept.remove(s);
                    break;
                }
            }
        }
        return kept;
    }

    /** Raw UTF-8 bytes of one token (the streaming decoder assembles code points across tokens). */
    public byte[] decodeTokenBytes(int token) {
        return tokenizer.decodeBytes(new int[] {token});
    }

    public String decode(int token) {
        return new String(decodeTokenBytes(token), StandardCharsets.UTF_8);
    }

    public String decode(IntSequence tokens) {
        return new String(tokenizer.decodeBytes(tokens), StandardCharsets.UTF_8);
    }
}
