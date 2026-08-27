package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jota.memory.MemoryView;
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
     * Bans the think markers so a non-thinking request can never open a reasoning span. No-op for
     * models without think markers.
     */
    static Sampler banMarkers(Sampler inner, Tokenizer tokenizer, String open, String close) {
        Integer thinkStart = boxed(SpecialTokens.find(tokenizer, open));
        Integer thinkEnd = boxed(SpecialTokens.find(tokenizer, close));
        Set<Integer> banned = new HashSet<>();
        if (thinkStart != null) banned.add(thinkStart);
        if (thinkEnd != null) banned.add(thinkEnd);
        return Sampler.banning(inner, banned);
    }

    /**
     * Caps the think span: once {@code budget} tokens have been sampled inside {@code <think>}, a
     * paragraph break and then the close marker are forced, so the remaining completion budget
     * always goes to content (thinking models otherwise starve the answer under tight max_tokens).
     * The break matters: a close forced MID-SENTENCE is off-training-distribution, and a small
     * model's continuation is then a fabricated turn header - a turn-guard stop and an EMPTY answer
     * (SmolLM3 bake-off, tight budget: hard close 2/6 empty, boundary-seeking close 6/6, paragraph
     * break 0/6). Cumulative across spans; forced tokens consume no RNG draw. Negative = uncapped.
     * {@code startInThink} starts INSIDE the think span - for templates whose generation prompt
     * opens {@code <think>} itself: the open token never passes through the sampler, so without
     * this the budget silently never arms and a long reasoning run can starve the visible answer to
     * LENGTH. A non-blank {@code message} is forced between the paragraph breaks when the budget
     * runs out - the model "deciding" to wrap up in its own words (llama.cpp's {@code
     * --reasoning-budget-message}), so the visible answer continues coherently instead of from an
     * unexplained stop. Encoding is the ordinary, non-special-aware path, so message text can never
     * inject a marker id; a tokenizer that cannot encode it closes hard.
     */
    static Sampler capBudget(
            Sampler inner,
            Tokenizer tokenizer,
            int budget,
            boolean startInThink,
            String message,
            String open,
            String close) {
        Integer openId = boxed(SpecialTokens.find(tokenizer, open));
        Integer closeId = boxed(SpecialTokens.find(tokenizer, close));
        if (budget < 0 || openId == null || closeId == null) {
            return inner;
        }
        int openToken = openId, closeToken = closeId;
        int[] filler = encode(tokenizer, fillerText(message));
        Sampler markersBanned = Sampler.banning(inner, Set.of(openId, closeId));
        return new Sampler() {
            boolean inThink = startInThink;
            int thought;
            int[] pending = new int[0]; // forced filler + close, one id per draw
            int pendingPos;

            @Override
            public int sampleToken(MemoryView<?> logits) {
                if (pendingPos < pending.length) {
                    return pending[pendingPos++];
                }
                if (inThink && thought >= budget) {
                    inThink = false;
                    if (filler.length > 0) {
                        pending = new int[filler.length + 1];
                        System.arraycopy(filler, 0, pending, 0, filler.length);
                        pending[pending.length - 1] = closeToken;
                        pendingPos = 1;
                        return pending[0];
                    }
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

    /**
     * What a spent budget forces before the close: a paragraph break, or the caller's message
     * wrapped in one on both sides. The breaks are unconditional - they are the
     * on-training-distribution boundary the bake-off proved, and nobody writing a message should
     * have to think about whitespace.
     */
    private static String fillerText(String message) {
        if (message == null || message.isBlank()) {
            return "\n\n";
        }
        return "\n\n" + message.strip() + "\n\n";
    }

    private static int[] encode(Tokenizer tokenizer, String text) {
        try {
            return tokenizer.encodeToArray(text);
        } catch (RuntimeException unsupported) {
            return new int[0];
        }
    }

    private static Integer boxed(OptionalInt id) {
        return id.isPresent() ? id.getAsInt() : null;
    }
}
