package com.qxotic.jinfer.llm;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Special-token views over a {@link Tokenizer}. Lookups here are SPECIALS-ONLY - a name resolves
 * only when the vocabulary marks it non-NORMAL, so a scaffold marker id can never alias a plain
 * vocab string (content can't mint what only these lookups hand out). Plain {@link
 * Tokenizer#encode} is the safe direction; {@link #encoder} is the one marked-unsafe path for
 * callers that author scaffold as text (the Jinja render, --raw-prompt).
 */
public final class SpecialTokens {

    /**
     * The model's stop-token set: {@code eosTokenId} (when {@code >= 0}) plus every {@code names}
     * entry the tokenizer actually has - absent names are skipped, so one list serves every
     * checkpoint of a family.
     *
     * <p>Insertion-ordered, EOS first. Generation tests every sampled token against the whole set
     * ({@code Generator}); the order matters for the other, rarer use - the two places that must
     * WRITE a terminator into the stream rather than recognize one (a grammar's dead end, a forced
     * call's end; {@code RequestPolicy.endTurn}). Any stop ends the turn, so the old hash order was
     * never wrong, just arbitrary: the emitted id could be some scaffold marker and could differ
     * between vocabularies. First-is-EOS makes it the model's own end-of-turn, always.
     */
    public static Set<Integer> stops(Tokenizer tokenizer, int eosTokenId, String... names) {
        Set<Integer> stops = new LinkedHashSet<>();
        if (eosTokenId >= 0) stops.add(eosTokenId);
        for (String name : names) find(tokenizer, name).ifPresent(stops::add);
        return stops;
    }

    private SpecialTokens() {}

    /** The id of {@code name} if it exists AND is a special token. */
    public static OptionalInt find(Tokenizer tokenizer, String name) {
        Vocabulary vocab = tokenizer.vocabulary();
        OptionalInt id = vocab.findId(name);
        return id.isPresent() && isSpecial(tokenizer, id.getAsInt()) ? id : OptionalInt.empty();
    }

    /** The id of the first present special among ordered alias spellings (e.g. bos/eos names). */
    static OptionalInt findFirst(Tokenizer tokenizer, String... names) {
        for (String name : names) {
            OptionalInt id = find(tokenizer, name);
            if (id.isPresent()) return id;
        }
        return OptionalInt.empty();
    }

    /**
     * The sequence-start spellings jinfer's families use, most specific first. ONE table, because
     * three call sites each carried their own two-name list ({@code "<bos>", "<|startoftext|>"})
     * and none of them knew Llama 3's {@code <|begin_of_text|>}. The cost was not cosmetic: the
     * Jinja whole-render binds {@code bos_token} from this lookup, and a null there renders the
     * literal four characters {@code None} at the very front of the prompt.
     *
     * <p>ponytail: a name table is a heuristic. The authority is the GGUF's {@code
     * tokenizer.ggml.bos_token_id}, which today only each port's Configuration reads - thread that
     * onto the tokenizer and this table becomes a fallback.
     */
    private static final String[] BOS_NAMES = {
        "<|begin_of_text|>", // Llama 3.x
        "<bos>", // Gemma
        "<s>", // Llama 2, Mistral
        "<|startoftext|>",
    };

    /**
     * As {@link #BOS_NAMES} for end-of-sequence, for the ONE caller that needs a single id (the
     * Jinja {@code eos_token} binding). Deliberately NOT the stop set: {@link #stops} collects
     * every terminator a family has - Llama 3 alone ends turns with {@code <|eot_id|>} and
     * tool-call messages with {@code <|eom_id|>} - and each port still declares its own list
     * because which tokens END GENERATION is a modelling fact, not a spelling.
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

    private static final ConcurrentHashMap<Tokenizer, int[]> newlines = new ConcurrentHashMap<>();

    /**
     * Token ids that decode to newlines only (LF/CR), per tokenizer (cached). Chat templates emit
     * {@code </think>\n} before the answer; grammar gating passes these through untouched so the
     * boilerplate newline is not consumed by a grammar with no whitespace in its language.
     */
    public static int[] newlineTokens(Tokenizer tokenizer) {
        return newlines.computeIfAbsent(
                tokenizer,
                t -> {
                    List<Integer> ids = new ArrayList<>();
                    for (int i = 0, n = t.vocabulary().size(); i < n; i++) {
                        byte[] b = t.decodeBytes(new int[] {i});
                        if (b.length == 0) continue;
                        boolean nl = true;
                        for (byte x : b) {
                            int c = x & 0xFF;
                            if (c != '\n' && c != '\r') {
                                nl = false;
                                break;
                            }
                        }
                        if (nl) ids.add(i);
                    }
                    int[] arr = new int[ids.size()];
                    for (int i = 0; i < arr.length; i++) arr[i] = ids.get(i);
                    return arr;
                });
    }
}
