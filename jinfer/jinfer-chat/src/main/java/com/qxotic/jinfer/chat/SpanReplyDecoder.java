package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.GgufTokenizer;
import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * The standard {@link ReplyDecoder}: routes each generated token to the content or reasoning
 * channel by the model's think markers, lets an optional {@link ToolCallDetector} claim call spans
 * (nothing from a span ever reaches a text channel), drops other scaffold specials, and UTF-8
 * decodes the rest incrementally (multi-byte sequences split across tokens are buffered, ids
 * tracked alongside bytes so every emitted fragment carries its verbatim payload ids).
 *
 * <p>One implementation serves every marker-structured model: templates compose it with their think
 * ids and a {@link SpanToolCallDetector} (or a hand-written detector for formats that are not flat
 * marker spans). Models whose reply grammar is entirely different (Harmony channels) implement
 * {@link ReplyDecoder} directly.
 */
public final class SpanReplyDecoder implements ReplyDecoder {

    private final GgufTokenizer tokenizer;
    private final int thinkOpen; // -1 when the model has no think channel
    private final int thinkClose;
    private final ToolCallDetector toolCalls; // null when the model has no native call format

    private final ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();
    private IntSequence.Builder pendingIds = IntSequence.newBuilder();
    private final CharsetDecoder utf8 =
            StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT);

    private boolean inThink;
    private boolean finished;
    private int seenCalls;

    private final PartsBuilder parts = new PartsBuilder(); // coalesced top-level accumulation
    private PartsBuilder reasoningContent; // the open think node's content, null when closed
    private IntSequence.Builder reasoningIds = IntSequence.newBuilder();

    /** Think markers resolved from the tokenizer's {@code <think>}/{@code </think>} specials. */
    public SpanReplyDecoder(GgufTokenizer tokenizer, ToolCallDetector toolCalls) {
        this.tokenizer = tokenizer;
        this.thinkOpen = tokenizer.getSpecialTokens().getOrDefault("<think>", -1);
        this.thinkClose = tokenizer.getSpecialTokens().getOrDefault("</think>", -1);
        this.toolCalls = toolCalls;
    }

    @Override
    public List<Part> feed(int token) {
        if (finished) throw new IllegalStateException("decoder already finished");
        List<Part> deltas = new ArrayList<>(1);
        // The detector claims call spans first, on token ids, so a call never reaches a text
        // channel (and never leaks mid-span). Channel transitions flush pending bytes so a split
        // sequence never bleeds across.
        if (toolCalls != null && toolCalls.accept(token)) {
            flushPending(deltas);
            collectNewCalls(deltas);
            return deltas;
        }
        if (thinkOpen >= 0 && token == thinkOpen) {
            flushPending(deltas);
            if (!inThink) {
                inThink = true;
                reasoningContent = new PartsBuilder();
                reasoningIds = IntSequence.newBuilder();
            }
            return deltas;
        }
        if (thinkClose >= 0 && token == thinkClose) {
            flushPending(deltas);
            if (inThink) {
                closeThink();
                // The span-close event: an empty Reasoning delta, so streaming consumers can
                // bracket the span exactly (finish() closes silently - an unterminated span has
                // no close marker to report).
                deltas.add(new Part.Reasoning(List.of(), IntSequence.empty()));
            }
            return deltas;
        }
        if (tokenizer.isSpecialToken(token)) {
            return deltas; // scaffold (stop / turn-close / channel ids): structural, dropped
        }
        pendingBytes.writeBytes(tokenizer.decodeTokenBytes(token));
        pendingIds.add(token);
        decodePending(deltas);
        return deltas;
    }

    @Override
    public List<Part> finish() {
        if (finished) return List.of();
        List<Part> deltas = new ArrayList<>(1);
        flushPending(deltas);
        closeThink(); // an unterminated think span is still reasoning
        finished = true;
        return deltas;
    }

    @Override
    public Message message() {
        if (!finished) throw new IllegalStateException("finish() the decoder first");
        return new Message(Role.ASSISTANT, parts.parts());
    }

    // ---- emission (delta + coalesced accumulation, one code path) ----

    private void emitFragment(List<Part> deltas, String fragment, IntSequence ids) {
        if (fragment.isEmpty()) return;
        if (inThink) {
            Part.Text text = new Part.Text(fragment, ids);
            deltas.add(new Part.Reasoning(List.of(text), ids));
            reasoningContent.text(fragment, ids);
            reasoningIds.addAll(ids);
        } else {
            deltas.add(new Part.Text(fragment, ids));
            parts.text(fragment, ids);
        }
    }

    /**
     * The coalesced accumulation for one channel: closed parts plus the currently open text run
     * (consecutive fragments append to a builder; the run materializes as ONE {@link Part.Text}
     * when a non-text part arrives or the channel closes - not rebuilt per token).
     */
    private static final class PartsBuilder {
        private final List<Part> parts = new ArrayList<>();
        private final StringBuilder text = new StringBuilder();
        private IntSequence.Builder ids = IntSequence.newBuilder();

        void text(String fragment, IntSequence fragmentIds) {
            text.append(fragment);
            ids.addAll(fragmentIds);
        }

        void add(Part part) {
            closeRun(); // a non-text part ends the run; later text starts a new one
            parts.add(part);
        }

        List<Part> parts() {
            closeRun();
            return parts;
        }

        private void closeRun() {
            if (!text.isEmpty()) {
                parts.add(new Part.Text(text.toString(), ids.build()));
                text.setLength(0);
                ids = IntSequence.newBuilder();
            }
        }
    }

    private void collectNewCalls(List<Part> deltas) {
        List<Part.ToolCall> all = toolCalls.calls();
        for (; seenCalls < all.size(); seenCalls++) {
            Part.ToolCall call = all.get(seenCalls);
            deltas.add(call);
            if (inThink) reasoningContent.add(call);
            else parts.add(call);
        }
    }

    private void closeThink() {
        if (!inThink) return;
        inThink = false;
        parts.add(new Part.Reasoning(reasoningContent.parts(), reasoningIds.build()));
        reasoningContent = null;
        reasoningIds = IntSequence.newBuilder();
    }

    // ---- incremental UTF-8 (ids tracked alongside bytes) ----

    private void decodePending(List<Part> deltas) {
        try {
            utf8.reset();
            CharBuffer chars = utf8.decode(ByteBuffer.wrap(pendingBytes.toByteArray()));
            if (!chars.isEmpty()) {
                emitFragment(deltas, chars.toString(), takePendingIds());
                pendingBytes.reset();
            }
        } catch (CharacterCodingException incomplete) {
            // Wait for a later token to complete a split UTF-8 sequence.
        }
    }

    private void flushPending(List<Part> deltas) {
        if (pendingBytes.size() > 0) {
            emitFragment(deltas, pendingBytes.toString(StandardCharsets.UTF_8), takePendingIds());
            pendingBytes.reset();
        }
    }

    private IntSequence takePendingIds() {
        IntSequence ids = pendingIds.build();
        pendingIds = IntSequence.newBuilder();
        return ids;
    }
}
