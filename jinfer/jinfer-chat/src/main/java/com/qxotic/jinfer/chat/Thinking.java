package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.HashSet;
import java.util.Set;

/**
 * Think-channel sampling controls: {@link Sampler} wrappers that steer the model's reasoning span
 * using its think-marker ids. Chat-layer knowledge (the markers are template structure), layered on
 * top of the token-level sampler by the generation driver.
 */
public final class Thinking {

    private Thinking() {}

    /**
     * Bans the {@code <think>}/{@code </think>} markers so a non-thinking request can never open a
     * reasoning span. No-op for models without think markers.
     */
    public static Sampler banMarkers(Sampler inner, Tokenizer tokenizer) {
        Integer thinkStart = boxed(SpecialTokens.find(tokenizer, "<think>"));
        Integer thinkEnd = boxed(SpecialTokens.find(tokenizer, "</think>"));
        Set<Integer> banned = new HashSet<>();
        if (thinkStart != null) banned.add(thinkStart);
        if (thinkEnd != null) banned.add(thinkEnd);
        return Sampler.banning(inner, banned);
    }

    /**
     * Caps the think span: once {@code budget} tokens have been sampled inside {@code <think>}, the
     * close marker is forced so the remaining completion budget always goes to content (thinking
     * models otherwise starve the answer under tight max_tokens). Cumulative across spans; the
     * forced token consumes no RNG draw. Negative = uncapped.
     */
    public static Sampler capBudget(Sampler inner, Tokenizer tokenizer, int budget) {
        Integer open = boxed(SpecialTokens.find(tokenizer, "<think>"));
        Integer close = boxed(SpecialTokens.find(tokenizer, "</think>"));
        if (budget < 0 || open == null || close == null) {
            return inner;
        }
        int openToken = open, closeToken = close;
        return new Sampler() {
            boolean inThink;
            int thought;

            @Override
            public int sampleToken(FloatTensor logits) {
                if (inThink && thought >= budget) {
                    inThink = false;
                    return closeToken;
                }
                int token = inner.sampleToken(logits);
                if (token == openToken) inThink = true;
                else if (token == closeToken) inThink = false;
                else if (inThink) thought++;
                return token;
            }
        };
    }

    private static Integer boxed(java.util.OptionalInt id) {
        return id.isPresent() ? id.getAsInt() : null;
    }
}
