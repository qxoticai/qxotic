package com.qxotic.jinfer.x.llm;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.HashSet;
import java.util.LinkedHashSet;
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

    /**
     * A family's stop set, insertion-ordered: {@code eosTokenId} first when present (the GGUF's own
     * end-of-sequence, added unchecked - the metadata id IS special by definition), then every
     * {@code names} spelling the vocabulary carries as a special. The ORDER is load-bearing: the
     * first element is the id a decode ended from outside emits (the model's own end-of-turn),
     * which {@code LoadedModel} preserves by refusing {@code Set.copyOf}.
     */
    public static Set<Integer> stops(Tokenizer tokenizer, int eosTokenId, String... names) {
        Set<Integer> stops = new LinkedHashSet<>();
        if (eosTokenId >= 0) stops.add(eosTokenId);
        for (String name : names) find(tokenizer, name).ifPresent(stops::add);
        return stops;
    }

    /**
     * Start-of-sequence spellings, in preference order, for the ONE caller that needs a single id
     * (the Jinja {@code bos_token} binding). A family that spells start differently declares it
     * onto the tokenizer and this table becomes a fallback.
     */
    private static final String[] BOS_NAMES = {
        "<|begin_of_text|>", // Llama 3.x
        "<bos>", // Gemma
        "<s>", // Llama 2, Mistral
        "<|startoftext|>",
    };

    /**
     * As {@link #BOS_NAMES} for end-of-sequence (the Jinja {@code eos_token} binding). Deliberately
     * NOT the stop set: which tokens END GENERATION is a modelling fact each port declares, not a
     * spelling.
     */
    private static final String[] EOS_NAMES = {
        "<|eot_id|>", // Llama 3.x turn end - what its tokenizer_config calls eos_token
        "<eos>", // Gemma
        "<|im_end|>", // Qwen and the ChatML families
        "<|end_of_text|>",
        "<|endoftext|>",
    };

    /** The vocabulary's sequence-start token, whatever this family spells it. */
    public static OptionalInt bos(Tokenizer tokenizer) {
        return findFirst(tokenizer, BOS_NAMES);
    }

    /** The vocabulary's end-of-sequence token, whatever this family spells it. */
    public static OptionalInt eos(Tokenizer tokenizer) {
        return findFirst(tokenizer, EOS_NAMES);
    }

    private static OptionalInt findFirst(Tokenizer tokenizer, String[] names) {
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

    /** Every spelling the vocabulary marks as non-normal. */
    public static Set<String> spellings(Tokenizer tokenizer) {
        Vocabulary vocab = tokenizer.vocabulary();
        Set<String> names = new HashSet<>();
        for (int id = 0; id < vocab.size(); id++) {
            if (isSpecial(tokenizer, id)) names.add(vocab.token(id));
        }
        return Set.copyOf(names);
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
        Set<String> names = spellings(tokenizer);
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
