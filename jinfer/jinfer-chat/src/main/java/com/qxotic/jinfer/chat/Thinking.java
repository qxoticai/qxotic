package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.HashSet;
import java.util.Set;

/**
 * Think-channel controls: {@link Sampler} wrappers that steer the model's reasoning span using its
 * think-marker ids, and {@link Inline}, which puts the markers back into a merged text stream.
 * Chat-layer knowledge (the markers are template structure), layered on top of the token-level
 * sampler by the generation driver.
 */
public final class Thinking {

    /** The think-span marker spellings - THE convention, spelled once. */
    static final String OPEN = "<think>";

    static final String CLOSE = "</think>";

    private Thinking() {}

    /**
     * Projects the two-channel reply (each fragment tagged reasoning or content) into ONE text
     * stream with the think span marked inline, for consumers that want thinking in the content
     * rather than in a channel of its own: llama.cpp's {@code reasoning_format: "none"} and the
     * CLI's {@code --think} display text.
     *
     * <p>Stateful per reply, because a fragment does not know what the previous one was: the open
     * marker attaches to the FIRST reasoning fragment and the close to the FIRST content fragment
     * after the span. An unterminated span (generation ended while thinking) stays unclosed, which
     * is what the raw token stream did.
     */
    public static final class Inline {

        private boolean open;

        /** The fragment as it should appear inline, markers attached at channel transitions. */
        public String project(String fragment, boolean reasoning) {
            if (reasoning) {
                if (!open) {
                    open = true;
                    return OPEN + fragment;
                }
                return fragment;
            }
            if (open) {
                open = false;
                return CLOSE + fragment;
            }
            return fragment;
        }
    }

    /**
     * Bans the {@code <think>}/{@code </think>} markers so a non-thinking request can never open a
     * reasoning span. No-op for models without think markers.
     */
    public static Sampler banMarkers(Sampler inner, Tokenizer tokenizer) {
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
     * the think span - for templates whose generation prompt opens {@code <think>} itself (Qwen3.5,
     * MiniCPM5, Nemotron): the open token never passes through the sampler, so without this the
     * budget silently never arms and a long reasoning run can starve the visible answer to LENGTH.
     */
    public static Sampler capBudget(
            Sampler inner, Tokenizer tokenizer, int budget, boolean startInThink) {
        Integer open = boxed(SpecialTokens.find(tokenizer, OPEN));
        Integer close = boxed(SpecialTokens.find(tokenizer, CLOSE));
        if (budget < 0 || open == null || close == null) {
            return inner;
        }
        int openToken = open, closeToken = close;
        return new Sampler() {
            boolean inThink = startInThink;
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
