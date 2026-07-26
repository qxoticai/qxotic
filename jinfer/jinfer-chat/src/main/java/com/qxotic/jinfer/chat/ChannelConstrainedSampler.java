package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntPredicate;

/**
 * Channel-scoped grammar constraint: the {@link ReplyParser} is the channel authority, and the
 * grammar exists only where the parser says text becomes output. Per position: an unmapped channel
 * (reasoning, structure, call payloads) samples free; a mapped channel masks logits to the
 * grammar's admissible set UNION the special tokens (by the two-domain law specials never carry
 * text - they are the model's right to close spans, open a think block, or end the turn). The
 * cursor advances only for plain tokens consumed into its channel, so grammar state can never
 * desync from the parser's routing (fragment emptiness under pending UTF-8 is irrelevant here).
 *
 * <p>Owns a PRIVATE parser instance (pre-fed the reply seed by the caller), fed every sampled token
 * - deterministic parsers guarantee it tracks the sink-side parser exactly. One newline tolerance
 * survives from the old gate: while a cursor has consumed nothing, newline tokens pass
 * unconstrained (the boilerplate between a span close and the answer).
 */
final class ChannelConstrainedSampler implements Sampler {

    private final Sampler inner;
    private final ReplyParser parser;
    private final Function<String, Grammar.Cursor> byChannel; // null = free channel
    private final IntPredicate isSpecial;
    private final int[] escapeIds; // span-openers re-allowed pre-start (the escape)
    private final int[] newlineIds;
    private final int stopToken;
    private final float[] savedEscape; // reusable saved-logit buffers (single-threaded sampler)
    private final float[] savedNewlines;
    private final Map<Grammar.Cursor, Boolean> started = new HashMap<>();
    private int newlineTolerance; // boilerplate newlines still admissible (armed by a special)
    private boolean escapeSpent; // a free region was visited: reasoning happened, escape retires

    ChannelConstrainedSampler(
            Sampler inner,
            ReplyParser parser,
            Function<String, Grammar.Cursor> byChannel,
            IntPredicate isSpecial,
            int[] escapeIds,
            int[] newlineIds,
            int stopToken) {
        this.inner = inner;
        this.parser = parser;
        this.byChannel = byChannel;
        this.isSpecial = isSpecial;
        this.escapeIds = escapeIds;
        this.newlineIds = newlineIds == null ? new int[0] : newlineIds;
        this.stopToken = stopToken;
        this.savedEscape = new float[this.escapeIds.length];
        this.savedNewlines = new float[this.newlineIds.length];
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        String channel = parser.pendingChannel();
        Grammar.Cursor cursor = channel == null ? null : byChannel.apply(channel);
        int token;
        boolean newlinePass = false;
        if (cursor == null) {
            escapeSpent = true; // reasoning/structure is underway: the open marker did its job
            token = inner.sampleToken(logits); // free channel: reasoning, structure, payloads
        } else {
            // BEFORE the grammar starts, two accommodations stay legal in the mask (restored
            // after maskLogits, so the inner samples exactly ONCE per position - a discarded
            // peek would burn an RNG draw and nudge stateful inners like capBudget):
            // - the escape hatch: the span-opening marker (the model's right to reason) - never
            //   the full special set: restoring stop tokens would let the model end the turn
            //   instead of complying. ONE-SHOT: once a free region was visited it retires, or
            //   an exhausted reasoning cap cycles open/force-close/newlines to the token budget
            //   (monotonic, like the old gate's phase machine)
            // - the boilerplate-newline tolerance: models are TRAINED to emit "</think>\n\n"
            //   before the answer, so directly after a structure token (a span close), up to
            //   TWO newlines pass without advancing the grammar - scoped (a cold start gets
            //   none: a newline-loving model must not free-run) and bounded by construction
            boolean preStart = !started.getOrDefault(cursor, false);
            int[] escape = preStart && !escapeSpent ? escapeIds : NONE;
            int[] tolerated = preStart && newlineTolerance > 0 ? newlineIds : NONE;
            save(logits, escape, savedEscape);
            save(logits, tolerated, savedNewlines);
            if (!cursor.maskLogits(logits)) {
                cursor.advanceWith(stopToken); // grammar complete or dead: end cleanly
                parser.feed(stopToken);
                return stopToken;
            }
            restore(logits, escape, savedEscape);
            restore(logits, tolerated, savedNewlines);
            token = inner.sampleToken(logits);
            newlinePass = tolerated.length > 0 && isNewline(token);
        }
        parser.feed(token);
        boolean special = isSpecial.test(token);
        // advance the owning cursor: plain token, mapped channel, not the newline tolerance -
        // the two-domain law makes "special => structure, plain => this channel's text" exact
        if (cursor != null && !newlinePass && !special) {
            cursor.advanceWith(token);
            started.put(cursor, true);
        }
        newlineTolerance = special ? 2 : newlinePass ? newlineTolerance - 1 : 0;
        return token;
    }

    private static final int[] NONE = new int[0];

    private static void save(FloatTensor logits, int[] ids, float[] saved) {
        for (int i = 0; i < ids.length; i++) saved[i] = logits.getFloat(ids[i]);
    }

    private static void restore(FloatTensor logits, int[] ids, float[] saved) {
        for (int i = 0; i < ids.length; i++) logits.setFloat(ids[i], saved[i]);
    }

    private boolean isNewline(int token) {
        for (int id : newlineIds) if (id == token) return true;
        return false;
    }
}
