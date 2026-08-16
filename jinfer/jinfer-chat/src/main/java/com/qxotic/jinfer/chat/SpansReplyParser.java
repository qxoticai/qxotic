package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Span-based content/reasoning parser used by marker-structured model families. */
final class SpansReplyParser implements ReplyParser {
    private final Tokenizer tokenizer;
    private final int thinkOpen;
    private final int thinkClose;
    private final SpanToolCallDetector toolCalls;
    private final boolean claimToolCalls;
    private final PendingUtf8 pending = new PendingUtf8();
    private final ContentBuilder content = new ContentBuilder();
    private ContentBuilder reasoningContent;
    private IntSequence.Builder reasoningIds = IntSequence.newBuilder();
    private Message result;
    private boolean generated;
    private boolean inThink;
    private boolean lastReasoning;
    private int seenSpans;

    SpansReplyParser(
            Tokenizer tokenizer,
            SpanToolCallDetector toolCalls,
            String thinkOpen,
            String thinkClose) {
        this(tokenizer, toolCalls, thinkOpen, thinkClose, true);
    }

    SpansReplyParser(
            Tokenizer tokenizer,
            SpanToolCallDetector toolCalls,
            String thinkOpen,
            String thinkClose,
            boolean claimToolCalls) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.toolCalls = toolCalls;
        this.claimToolCalls = claimToolCalls;
        this.thinkOpen = SpecialTokens.find(tokenizer, thinkOpen).orElse(-1);
        this.thinkClose = SpecialTokens.find(tokenizer, thinkClose).orElse(-1);
    }

    @Override
    public void seed(IntSequence seed) {
        if (generated) throw new IllegalStateException("cannot seed after generated tokens");
        if (result != null) throw new IllegalStateException("parser already finished");
        seed.forEachInt(this::consume);
        pending.flush();
        content.reset();
        reasoningIds = IntSequence.newBuilder();
        if (reasoningContent != null) reasoningContent.reset();
        if (toolCalls != null) seenSpans = toolCalls.spans().size();
        lastReasoning = false;
    }

    @Override
    public Fragment feed(int token) {
        if (result != null) throw new IllegalStateException("parser already finished");
        generated = true;
        return consume(token);
    }

    private Fragment consume(int token) {
        if (toolCalls != null && toolCalls.accept(token)) {
            Fragment flushed = flushPending();
            return concat(flushed, collectSpans());
        }
        if (thinkOpen >= 0 && token == thinkOpen) {
            Fragment flushed = flushPending();
            if (!inThink) {
                inThink = true;
                reasoningContent = new ContentBuilder();
                reasoningIds = IntSequence.newBuilder();
            }
            return flushed;
        }
        if (thinkClose >= 0 && token == thinkClose) {
            Fragment flushed = flushPending();
            closeThink();
            return flushed;
        }
        if (SpecialTokens.isSpecial(tokenizer, token)) return Fragment.EMPTY;
        PendingUtf8.Fragment fragment =
                pending.add(tokenizer.decodeBytes(new int[] {token}), token);
        return fragment == null ? Fragment.EMPTY : emit(fragment.text(), fragment.ids());
    }

    private static Fragment concat(Fragment a, Fragment b) {
        if (a.text().isEmpty()) return b;
        if (b.text().isEmpty()) return a;
        return new Fragment(a.text() + b.text(), a.tokens().concat(b.tokens()));
    }

    @Override
    public boolean reasoning() {
        return lastReasoning;
    }

    @Override
    public Channel channel() {
        if (claimToolCalls && toolCalls != null && toolCalls.inSpan()) return Channel.TOOL_CALL;
        return inThink ? Channel.REASONING : Channel.CONTENT;
    }

    @Override
    public Channel pending() {
        // the marker world nests for real: a claimed call span can sit inside an open think span
        return channel() == Channel.TOOL_CALL && inThink ? Channel.REASONING : null;
    }

    @Override
    public Set<Channel> outputChannels() {
        return Set.of(Channel.CONTENT);
    }

    @Override
    public Message finish() {
        if (result == null) {
            flushPending();
            closeThink();
            result = new Message(Role.ASSISTANT, content.parts());
        }
        return result;
    }

    private Fragment emit(String fragment, IntSequence ids) {
        if (fragment.isEmpty()) return Fragment.EMPTY;
        lastReasoning = inThink;
        if (inThink) {
            reasoningContent.text(fragment, ids);
            reasoningIds.addAll(ids);
        } else {
            content.text(fragment, ids);
        }
        return new Fragment(fragment, ids);
    }

    private Fragment collectSpans() {
        StringBuilder visible = new StringBuilder();
        IntSequence.Builder visibleIds = IntSequence.newBuilder();
        List<SpanToolCallDetector.Span> spans = toolCalls.spans();
        for (; seenSpans < spans.size(); seenSpans++) {
            SpanToolCallDetector.Span span = spans.get(seenSpans);
            ContentBuilder target = inThink ? reasoningContent : content;
            if (claimToolCalls && !span.calls().isEmpty()) {
                for (Content.ToolCall call : span.calls()) target.add(call);
            } else {
                target.add(new Content.Text(span.text(), span.wireIds()));
                visible.append(span.text());
                visibleIds.addAll(span.wireIds());
                lastReasoning = inThink;
            }
        }
        return visible.isEmpty()
                ? Fragment.EMPTY
                : new Fragment(visible.toString(), visibleIds.build());
    }

    private void closeThink() {
        if (!inThink) return;
        inThink = false;
        content.add(new Content.Reasoning(reasoningContent.parts(), reasoningIds.build()));
        reasoningContent = null;
        reasoningIds = IntSequence.newBuilder();
    }

    private Fragment flushPending() {
        PendingUtf8.Fragment fragment = pending.flush();
        return fragment == null ? Fragment.EMPTY : emit(fragment.text(), fragment.ids());
    }

    private static final class ContentBuilder {
        private final List<Content> parts = new ArrayList<>();
        private final StringBuilder text = new StringBuilder();
        private IntSequence.Builder ids = IntSequence.newBuilder();

        void reset() {
            parts.clear();
            text.setLength(0);
            ids = IntSequence.newBuilder();
        }

        void text(String value, IntSequence verbatim) {
            text.append(value);
            ids.addAll(verbatim);
        }

        void add(Content part) {
            closeText();
            parts.add(part);
        }

        List<Content> parts() {
            closeText();
            return List.copyOf(parts);
        }

        private void closeText() {
            if (text.isEmpty()) return;
            parts.add(new Content.Text(text.toString(), ids.build()));
            text.setLength(0);
            ids = IntSequence.newBuilder();
        }
    }
}
