package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * The decode direction of a {@link ChatTemplate}: a stateful, single-use parser over one generated
 * reply token stream. Feed EVERY sampled token in order, the trailing stop token included
 * (recognized scaffold specials are inert). The parser states grammar facts; what to SHOW is the
 * caller's policy.
 *
 * <p>{@link #feed} returns the displayable text fragment the token completed - {@code ""} while
 * there is nothing to show (scaffold, a claimed tool-call span, or a code point still spanning
 * tokens). Fragments are UTF-8 safe and never contain call syntax; {@link #reasoning()} is the
 * channel of the last non-empty fragment. Tool calls are ATOMIC: nothing surfaces mid-span, parsed
 * calls appear only in {@link #finish()}'s message (a span the generation never closed is no call).
 *
 * <p>Stop STRINGS stay outside the parser - the caller applies its stop-aware holdback to content
 * fragments and aborts the token loop; the parser is purely structural.
 */
public interface ReplyParser {

    /**
     * Consume the next generated token; returns the text fragment it completed, or {@code ""} when
     * there is nothing to show.
     */
    String feed(int token);

    /** Whether the last non-empty {@link #feed} fragment belongs to the reasoning channel. */
    boolean reasoning();

    /**
     * The channel a TEXT token would join if fed now, in the FAMILY'S OWN naming
     * ("reasoning"/"content" for span parsers; "analysis"/"commentary"/"final"/"tool-call" for
     * Harmony); {@code null} = a structure region (headers, span markers) where every token is
     * scaffold. The channel authority for channel-scoped constraints: reasoning and structure stay
     * free, {@link #outputChannels} carry the grammar.
     */
    String pendingChannel();

    /**
     * The channels whose text lands in the final output (what reaches the reply's content) - the
     * default target of an output grammar.
     */
    Set<String> outputChannels();

    /**
     * Flush and close open spans (an unterminated think span is still reasoning), then the
     * structured reply: coalesced text, the reasoning tree, tool calls - each model-produced part
     * carrying its verbatim payload ids. Role is always assistant. Idempotent.
     */
    Message finish();

    /**
     * Prompt bytes are not reply bytes: called once after the prompt's reply seed has been {@link
     * #feed}ed and before the first generated token, this drops any think/content TEXT the seed
     * accumulated while keeping the parse STATE it established (an open span stays open) - a
     * non-thinking scaffold's {@code </think>\n\n} tail must not surface as the reply's leading
     * newlines. A forced-call seed is the deliberate exception and survives: the seeded call
     * structure parses whole, so implementations leave call-region capture untouched.
     */
    default void beginReply() {}

    /**
     * The reply is structurally OVER - a parser that enforces its family's reply grammar reports
     * true once no further token can extend the reply (the control rule fired on an off-language
     * token, or the language accepted with nothing left to admit). Every later {@link #feed} is
     * inert, so a generation still running is burning budget on tokens that can never surface; the
     * driver treats this as the model's own end of turn. Grammarless parsers never end.
     */
    default boolean ended() {
        return false;
    }

    /** One-shot parse of a finished reply (trailing stop token included or not - both work). */
    static Message parse(ReplyParser parser, IntSequence reply) {
        reply.forEachInt(parser::feed);
        return parser.finish();
    }

    /**
     * The built-in span grammar (what marker-structured models use): content vs {@code
     * <think>}/{@code </think>} spans resolved from the vocabulary, no native tool-call format.
     */
    static ReplyParser spans(Tokenizer tokenizer) {
        return new SpansReplyParser(tokenizer, null);
    }

    /**
     * The span grammar plus a tool-call span: calls are claimed between the two named trusted
     * specials and their payload text parsed by {@code payload} (see {@link
     * ToolCallSyntax#parseBlock}). Models whose reply grammar is not span-shaped (Harmony channel
     * headers) implement this interface directly.
     */
    static ReplyParser spans(
            Tokenizer tokenizer,
            String callStart,
            String callEnd,
            Function<String, List<Part.ToolCall>> payload) {
        return spans(tokenizer, callStart, callEnd, payload, Thinking.OPEN, Thinking.CLOSE);
    }

    /**
     * As {@link #spans(Tokenizer, String, String, Function)}, with the reasoning span's markers
     * named by the family - for models that spell reasoning as a channel rather than {@code
     * <think>} (Gemma 4's {@code <|channel>thought}).
     */
    static ReplyParser spans(
            Tokenizer tokenizer,
            String callStart,
            String callEnd,
            Function<String, List<Part.ToolCall>> payload,
            String thinkOpen,
            String thinkClose) {
        return new SpansReplyParser(
                tokenizer,
                new SpanToolCallDetector(tokenizer, callStart, callEnd, payload),
                thinkOpen,
                thinkClose);
    }

    /**
     * {@code inner} with one token id silenced - for GGUFs that mistype a control token as NORMAL
     * (Gemma 4's {@code <eos>}), whose spelling would otherwise leak into content as literal text.
     */
    static ReplyParser dropping(ReplyParser inner, int id) {
        return new ReplyParser() {
            @Override
            public String feed(int token) {
                return token == id ? "" : inner.feed(token);
            }

            @Override
            public boolean reasoning() {
                return inner.reasoning();
            }

            @Override
            public String pendingChannel() {
                return inner.pendingChannel();
            }

            @Override
            public Set<String> outputChannels() {
                return inner.outputChannels();
            }

            @Override
            public Message finish() {
                return inner.finish();
            }

            @Override
            public void beginReply() {
                inner.beginReply();
            }

            @Override
            public boolean ended() {
                return inner.ended();
            }
        };
    }
}
