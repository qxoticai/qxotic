package com.qxotic.jinfer.llm;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.HashSet;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Special-token views over a {@link Tokenizer}. Lookups here are SPECIALS-ONLY - a name resolves
 * only when the vocabulary marks it non-NORMAL, so a scaffold marker id can never alias a plain
 * vocab string (content can't mint what only these lookups hand out). Plain {@link
 * Tokenizer#encode} is the safe direction; {@link #encoder} is the one marked-unsafe path for
 * callers that author scaffold as text (the Jinja render, --raw-prompt).
 */
public final class SpecialTokens {

    private SpecialTokens() {}

    /** The id of {@code name} if it exists AND is a special token. */
    public static OptionalInt find(Tokenizer tokenizer, String name) {
        Vocabulary vocab = tokenizer.vocabulary();
        OptionalInt id = vocab.findId(name);
        return id.isPresent() && isSpecial(tokenizer, id.getAsInt()) ? id : OptionalInt.empty();
    }

    /** The id of the first present special among ordered alias spellings (e.g. bos/eos names). */
    public static OptionalInt findFirst(Tokenizer tokenizer, String... names) {
        for (String name : names) {
            OptionalInt id = find(tokenizer, name);
            if (id.isPresent()) return id;
        }
        return OptionalInt.empty();
    }

    /** The id of a special token a curated template requires; throws naming it when absent. */
    public static int require(Tokenizer tokenizer, String name) {
        return find(tokenizer, name)
                .orElseThrow(() -> new IllegalArgumentException("tokenizer lacks " + name));
    }

    public static boolean isSpecial(Tokenizer tokenizer, int token) {
        Vocabulary vocab = tokenizer.vocabulary();
        return vocab.contains(token) && !vocab.isTokenOfType(token, StandardTokenType.NORMAL);
    }

    /**
     * A specials-aware encoder: maps special-token STRINGS in the text to their ids (plain {@code
     * encode} never does). Compile once and reuse - callers own the lifecycle.
     *
     * <p>toknroll's matcher rejects a set where one token is a strict prefix of another (e.g.
     * MiniCPM's {@code <param} vs {@code <parameters>}). Drop the shorter token, which under
     * longest-match is only reachable as a substring of the longer one - rendered control-token
     * streams never emit it standalone, so this is loss-free in practice.
     */
    public static Specials encoder(Tokenizer tokenizer) {
        Vocabulary vocab = tokenizer.vocabulary();
        Set<String> names = new HashSet<>();
        for (int id = 0; id < vocab.size(); id++) {
            if (isSpecial(tokenizer, id)) names.add(vocab.token(id));
        }
        Set<String> kept = new HashSet<>(names);
        for (String s : names) {
            for (String o : names) {
                if (!s.equals(o) && o.startsWith(s)) {
                    kept.remove(s);
                    break;
                }
            }
        }
        return Specials.compile(vocab, kept);
    }

    /** One-shot {@link #encoder} encode; prefer holding the encoder on hot paths. */
    public static IntSequence encode(Tokenizer tokenizer, String text) {
        return encoder(tokenizer).encode(tokenizer, text);
    }
}
