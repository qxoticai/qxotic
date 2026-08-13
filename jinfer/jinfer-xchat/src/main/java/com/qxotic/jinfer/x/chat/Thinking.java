package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.HashSet;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Think-channel controls: {@link Sampler} wrappers that steer the model's reasoning span using its
 * think-marker ids. Chat-layer knowledge (the markers are template structure), layered on top of
 * the token-level sampler by the engine's prepare.
 */
final class Thinking {

    /** The think-span marker spellings - THE convention, spelled once. */
    static final String OPEN = "<think>";

    static final String CLOSE = "</think>";

    private Thinking() {}

    /**
     * Bans the {@code <think>}/{@code </think>} markers so a non-thinking request can never open a
     * reasoning span. No-op for models without think markers.
     */
    static Sampler banMarkers(Sampler inner, Tokenizer tokenizer) {
        Integer thinkStart = boxed(SpecialTokens.find(tokenizer, OPEN));
        Integer thinkEnd = boxed(SpecialTokens.find(tokenizer, CLOSE));
        Set<Integer> banned = new HashSet<>();
        if (thinkStart != null) banned.add(thinkStart);
        if (thinkEnd != null) banned.add(thinkEnd);
        return Sampler.banning(inner, banned);
    }

    /**
     * Caps the think span: once {@code budget} tokens have been sampled inside {@code <think>}, the
     * close marker is forced so the remaining completion budget always goes to content (thinking
     * models otherwise starve the answer under tight max_tokens). Cumulative across spans; the
     * forced token consumes no RNG draw. Negative = uncapped. {@code startInThink} starts INSIDE
     * the think span - for templates whose generation prompt opens {@code <think>} itself: the open
     * token never passes through the sampler, so without this the budget silently never arms and a
     * long reasoning run can starve the visible answer to LENGTH.
     */
    static Sampler capBudget(Sampler inner, Tokenizer tokenizer, int budget, boolean startInThink) {
        Integer open = boxed(SpecialTokens.find(tokenizer, OPEN));
        Integer close = boxed(SpecialTokens.find(tokenizer, CLOSE));
        if (budget < 0 || open == null || close == null) {
            return inner;
        }
        int openToken = open, closeToken = close;
        Sampler markersBanned = Sampler.banning(inner, Set.of(open, close));
        return new Sampler() {
            boolean inThink = startInThink;
            int thought;

            @Override
            public int sampleToken(com.qxotic.jota.memory.MemoryView<?> logits) {
                if (inThink && thought >= budget) {
                    inThink = false;
                    return closeToken;
                }
                // a SPENT budget bans BOTH markers: a model force-closed mid-thought
                // re-opens on its very next token (greedy Qwen3.5 does, deterministically),
                // and with only the open banned its next-best is the PAIRED CLOSE - either
                // marker after the spend is scaffold the reply grammar does not expect, and
                // the un-banned cap ping-pongs marker noise until LENGTH with a blank
                // visible answer, the exact starvation the cap exists to prevent
                int token =
                        thought >= budget
                                ? markersBanned.sampleToken(logits)
                                : inner.sampleToken(logits);
                if (token == openToken) inThink = true;
                else if (token == closeToken) inThink = false;
                else if (inThink) thought++;
                return token;
            }
        };
    }

    private static Integer boxed(OptionalInt id) {
        return id.isPresent() ? id.getAsInt() : null;
    }
}
