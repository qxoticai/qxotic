package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * The built-in {@link ReplyParser}: routes each generated token to the content or reasoning channel
 * by the model's think markers, lets an optional {@link ToolCallDetector} claim call spans (nothing
 * from a span ever reaches a text channel), drops other scaffold specials, and UTF-8 decodes the
 * rest incrementally - multi-byte sequences split across tokens are buffered, ids tracked alongside
 * bytes so every accumulated part carries its verbatim payload ids.
 */
final class SpansReplyParser implements ReplyParser {

    private final Tokenizer tokenizer;
    private final int thinkOpen; // -1 when the model has no think channel
    private final int thinkClose;
    private final ToolCallDetector toolCalls; // null when the model has no native call format

    private final PendingUtf8 pending = new PendingUtf8();

    private boolean inThink;
    private boolean lastReasoning;
    private Message message; // built by finish(); non-null = finished
    private int seenCalls;

    private final PartsBuilder parts = new PartsBuilder(); // coalesced top-level accumulation
    private PartsBuilder reasoningContent; // the open think node's content, null when closed
    private IntSequence.Builder reasoningIds = IntSequence.newBuilder();

    /** Think markers resolved from the tokenizer's {@code <think>}/{@code </think>} specials. */
    SpansReplyParser(Tokenizer tokenizer, ToolCallDetector toolCalls) {
        this.tokenizer = tokenizer;
        this.thinkOpen = SpecialTokens.find(tokenizer, "<think>").orElse(-1);
        this.thinkClose = SpecialTokens.find(tokenizer, "</think>").orElse(-1);
        this.toolCalls = toolCalls;
    }

    @Override
    public String feed(int token) {
        if (message != null) throw new IllegalStateException("parser already finished");
        // The detector claims call spans first, on token ids, so a call never reaches a text
        // channel (and never leaks mid-span). Channel transitions flush pending bytes so a split
        // sequence never bleeds across; the flushed fragment belongs to the channel it was
        // accumulated on (the marker token itself contributes no text).
        if (toolCalls != null && toolCalls.accept(token)) {
            String flushed = flushPending();
            collectNewCalls();
            return flushed;
        }
        if (thinkOpen >= 0 && token == thinkOpen) {
            String flushed = flushPending();
            if (!inThink) {
                inThink = true;
                reasoningContent = new PartsBuilder();
                reasoningIds = IntSequence.newBuilder();
            }
            return flushed;
        }
        if (thinkClose >= 0 && token == thinkClose) {
            String flushed = flushPending();
            closeThink();
            return flushed;
        }
        if (SpecialTokens.isSpecial(tokenizer, token)) {
            return ""; // scaffold (stop / turn-close / channel ids): structural, nothing to show
        }
        PendingUtf8.Fragment fragment =
                pending.add(tokenizer.decodeBytes(new int[] {token}), token);
        return fragment == null ? "" : emitFragment(fragment.text(), fragment.ids());
    }

    @Override
    public boolean reasoning() {
        return lastReasoning;
    }

    @Override
    public Message finish() {
        if (message == null) {
            flushPending(); // a trailing split code point still lands in the message
            closeThink(); // an unterminated think span is still reasoning
            message = new Message(Role.ASSISTANT, parts.parts());
        }
        return message;
    }

    // ---- emission (fragment + coalesced accumulation, one code path) ----

    private String emitFragment(String fragment, IntSequence ids) {
        if (fragment.isEmpty()) return "";
        lastReasoning = inThink;
        if (inThink) {
            reasoningContent.text(fragment, ids);
            reasoningIds.addAll(ids);
        } else {
            parts.text(fragment, ids);
        }
        return fragment;
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

    private void collectNewCalls() {
        List<Part.ToolCall> all = toolCalls.calls();
        for (; seenCalls < all.size(); seenCalls++) {
            Part.ToolCall call = all.get(seenCalls);
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

    private String flushPending() {
        PendingUtf8.Fragment fragment = pending.flush();
        return fragment == null ? "" : emitFragment(fragment.text(), fragment.ids());
    }
}
