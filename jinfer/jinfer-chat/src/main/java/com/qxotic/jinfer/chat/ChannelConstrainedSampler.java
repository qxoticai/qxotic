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
            token = -1;
            // the boilerplate-newline tolerance: models are TRAINED to emit "</think>\n\n"
            // before the answer, so directly after a structure token (a span close), before the
            // grammar starts, up to TWO newlines pass unconstrained - scoped (a cold start gets
            // none: a newline-loving model must not free-run) and bounded by construction
            if (newlineTolerance > 0 && !started.getOrDefault(cursor, false)) {
                int peek = inner.sampleToken(logits); // unconstrained peek, grammar untouched
                if (isNewline(peek)) {
                    token = peek;
                    newlinePass = true;
                }
            }
            if (token < 0) {
                // the escape hatch: BEFORE the grammar starts, span-opening markers stay legal
                // (the model's right to reason) - never the full special set: restoring stop
                // tokens would let the model end the turn instead of complying. ONE-SHOT: once
                // a free region was visited the escape retires, or an exhausted reasoning cap
                // cycles open/force-close/newlines to the token budget (monotonic, like the old
                // gate's phase machine)
                boolean escape = !escapeSpent && !started.getOrDefault(cursor, false);
                float[] saved = escape ? new float[escapeIds.length] : null;
                if (escape) {
                    for (int i = 0; i < escapeIds.length; i++) {
                        saved[i] = logits.getFloat(escapeIds[i]);
                    }
                }
                if (!cursor.maskLogits(logits)) {
                    cursor.advanceWith(stopToken); // grammar complete or dead: end cleanly
                    parser.feed(stopToken);
                    return stopToken;
                }
                if (escape) {
                    for (int i = 0; i < escapeIds.length; i++) {
                        logits.setFloat(escapeIds[i], saved[i]);
                    }
                }
                token = inner.sampleToken(logits);
            }
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

    private boolean isNewline(int token) {
        for (int id : newlineIds) if (id == token) return true;
        return false;
    }
}
