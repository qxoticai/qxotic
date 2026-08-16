package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/** Stateful, single-use decoding of one generated reply. */
public interface ReplyParser {
    /** Applies prompt-owned reply tokens without exposing them as generated content. */
    void seed(IntSequence promptOwnedTokens);

    /** Consumes one generated token and returns its displayable UTF-8-safe fragment. */
    Fragment feed(int token);

    /**
     * One displayable fragment and the verbatim tokens that produced it: text is UTF-8-safe
     * (partial codepoints hold back), tokens re-encode it exactly. {@link #EMPTY} for structural
     * tokens (markers, scaffold) that surface nothing.
     */
    record Fragment(String text, IntSequence tokens) {
        static final Fragment EMPTY = new Fragment("", IntSequence.empty());
    }

    boolean reasoning();

    /** The innermost channel the next text token would join; null in structural regions. */
    Channel channel();

    /**
     * The channel suspended underneath {@link #channel()}, if any: a claimed call span opened while
     * a think span is still open reports {@link Channel#TOOL_CALL} with {@link Channel#REASONING}
     * pending. Nesting never exceeds one level, and grammar-driven parsers sequence regions instead
     * of suspending them, so they always report null - flat by construction.
     */
    Channel pending();

    Set<Channel> outputChannels();

    default boolean ended() {
        return false;
    }

    /** Finishes the structured assistant message. Idempotent. */
    Message finish();

    static Message parse(ReplyParser parser, IntSequence reply) {
        reply.forEachInt(parser::feed);
        return parser.finish();
    }

    static ReplyParser spans(Tokenizer tokenizer) {
        return new SpansReplyParser(tokenizer, null, "<think>", "</think>");
    }

    static ReplyParser spans(Tokenizer tokenizer, String thinkOpen, String thinkClose) {
        return new SpansReplyParser(tokenizer, null, thinkOpen, thinkClose);
    }

    static ReplyParser spans(
            Tokenizer tokenizer,
            String callStart,
            String callEnd,
            Function<String, List<Content.ToolCall>> payload) {
        return spans(tokenizer, callStart, callEnd, payload, "<think>", "</think>");
    }

    static ReplyParser spans(
            Tokenizer tokenizer,
            String callStart,
            String callEnd,
            Function<String, List<Content.ToolCall>> payload,
            String thinkOpen,
            String thinkClose) {
        return spans(tokenizer, callStart, callEnd, payload, thinkOpen, thinkClose, true);
    }

    /**
     * A span parser whose call syntax is either claimed structurally or retained as visible text.
     * Unclaimed spans keep their complete wire ids for exact re-encoding.
     */
    static ReplyParser spans(
            Tokenizer tokenizer,
            String callStart,
            String callEnd,
            Function<String, List<Content.ToolCall>> payload,
            String thinkOpen,
            String thinkClose,
            boolean claimToolCalls) {
        return new SpansReplyParser(
                tokenizer,
                new SpanToolCallDetector(tokenizer, callStart, callEnd, payload),
                thinkOpen,
                thinkClose,
                claimToolCalls);
    }

    static ReplyParser dropping(ReplyParser inner, int token) {
        Objects.requireNonNull(inner, "inner");
        return new ReplyParser() {
            private boolean generated;
            private boolean finished;

            public void seed(IntSequence seed) {
                if (generated)
                    throw new IllegalStateException("cannot seed after generated tokens");
                if (finished) throw new IllegalStateException("parser already finished");
                IntSequence.Builder filtered = IntSequence.newBuilder();
                Objects.requireNonNull(seed, "seed")
                        .forEachInt(
                                next -> {
                                    if (next != token) filtered.add(next);
                                });
                inner.seed(filtered.build());
            }

            public Fragment feed(int next) {
                if (finished) throw new IllegalStateException("parser already finished");
                generated = true;
                return next == token ? Fragment.EMPTY : inner.feed(next);
            }

            public boolean reasoning() {
                return inner.reasoning();
            }

            public Channel channel() {
                return inner.channel();
            }

            public Channel pending() {
                return inner.pending();
            }

            public Set<Channel> outputChannels() {
                return inner.outputChannels();
            }

            public boolean ended() {
                return inner.ended();
            }

            public Message finish() {
                finished = true;
                return inner.finish();
            }
        };
    }
}
