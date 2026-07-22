package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
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
     * Flush and close open spans (an unterminated think span is still reasoning), then the
     * structured reply: coalesced text, the reasoning tree, tool calls - each model-produced part
     * carrying its verbatim payload ids. Role is always assistant. Idempotent.
     */
    Message finish();

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
        return new SpansReplyParser(
                tokenizer, new SpanToolCallDetector(tokenizer, callStart, callEnd, payload));
    }
}
