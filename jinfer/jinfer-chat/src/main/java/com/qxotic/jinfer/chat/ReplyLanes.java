package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The reply's two text lanes, token by token: the native {@link ReplyParser}'s (content vs
 * reasoning) when the model has a codec, else raw decoded text. Owns the ONE parser instance of a
 * generation - pre-fed the forced-call seed so a seeded reply parses whole - and {@link #finish}es
 * the structured message from exactly the parse that streamed; there is no second decode pass, so
 * the streamed fragments and the final message can never disagree.
 */
public final class ReplyLanes {

    private final ReplyParser parser; // null: raw text, single lane
    private final boolean claimToolCalls;
    private final PendingUtf8 pending = new PendingUtf8();
    private final StringBuilder rawText;
    private final Tokenizer tokenizer;
    private boolean reasoning;

    /**
     * {@code claimToolCalls} false leaves call syntax the model emits on its own as visible TEXT:
     * the FAMILY's parser still structures the reply (its channels and reasoning markers are the
     * family's, not the generic span grammar's - swapping parsers would leak a Harmony analysis
     * channel straight into content), but call-span payloads stream as text and {@link #finish}
     * downgrades parsed calls to text parts.
     *
     * <p>That is the right reading when the caller offered no tools: a claimed call the client
     * never asked for is not a call, it is an answer the client cannot see. Models do emit it -
     * LFM2.5 answers a bare prompt with its own final_output call - and claiming it turns a plain
     * reply into a tool-call response with no text at all.
     */
    public ReplyLanes(
            Optional<ChatTemplate> template,
            Tokenizer tokenizer,
            int[] parserSeed,
            boolean claimToolCalls) {
        this.parser = template.map(ChatTemplate::parser).orElse(null);
        this.claimToolCalls = claimToolCalls;
        this.rawText = parser == null ? new StringBuilder() : null;
        this.tokenizer = tokenizer;
        if (parser != null) {
            for (int token : parserSeed) parser.feed(token);
        }
    }

    /** The text this token adds ("" while pending); {@link #reasoning()} tells its lane. */
    public String feed(int token) {
        if (parser == null) {
            String fragment = added(token);
            rawText.append(fragment);
            return fragment;
        }
        boolean inCall = !claimToolCalls && "tool-call".equals(parser.pendingChannel());
        String fragment = parser.feed(token);
        reasoning = parser.reasoning();
        if (fragment.isEmpty() && inCall && "tool-call".equals(parser.pendingChannel())) {
            // an unclaimed call-span PAYLOAD token (in-span before AND after the feed, so the
            // markers themselves stay silent): surface the raw text the parser withheld. It joins
            // the surrounding span's lane, and it matches finish()'s downgraded call byte-exactly
            // (a claimed call's verbatim is the payload ids, markers excluded).
            return added(token);
        }
        return fragment;
    }

    /**
     * The text {@code token} completes, or "" while a multi-byte sequence is still arriving.
     *
     * <p>{@link PendingUtf8#add} returns null for exactly that "not yet" case - a token carrying
     * the first half of a multi-byte character, which is ordinary for byte-level BPE. Both call
     * sites used to dereference it, so such a token crashed the pass with a NullPointerException
     * and the client got 500 "Internal server error" for a perfectly good request. It went
     * unnoticed because the native-codec path routes through the parser, which handles pending
     * itself: only the codec-less lane - which is what /v1/completions always uses - could reach
     * it, and only on a model that splits a character across tokens.
     */
    private String added(int token) {
        PendingUtf8.Fragment fragment =
                pending.add(tokenizer.decodeBytes(new int[] {token}), token);
        return fragment == null ? "" : fragment.text();
    }

    /** The lane of the LAST {@link #feed}ed fragment. */
    public boolean reasoning() {
        return reasoning;
    }

    /** The finished structured reply, from the same parse that streamed. */
    public Message finish() {
        if (parser == null) {
            // a reply that ENDS mid-sequence leaves bytes buffered; without this drain they were
            // simply lost from the finished message (flush decodes permissively, so a genuinely
            // truncated character becomes U+FFFD rather than vanishing)
            PendingUtf8.Fragment tail = pending.flush();
            if (tail != null) rawText.append(tail.text());
            return new Message(Role.ASSISTANT, rawText.toString());
        }
        Message reply = parser.finish();
        return claimToolCalls ? reply : declaimed(reply);
    }

    /** The reply with every parsed call downgraded to the text the model actually emitted. */
    private Message declaimed(Message reply) {
        return new Message(reply.role(), declaimed(reply.content()));
    }

    private List<Part> declaimed(List<Part> parts) {
        List<Part> out = new ArrayList<>(parts.size());
        for (Part part : parts) {
            switch (part) {
                case Part.ToolCall call -> out.add(new Part.Text(callText(call), call.verbatim()));
                case Part.Reasoning r ->
                        out.add(new Part.Reasoning(declaimed(r.content()), r.verbatim()));
                default -> out.add(part);
            }
        }
        return out;
    }

    private String callText(Part.ToolCall call) {
        if (call.verbatim() != null) return tokenizer.decode(call.verbatim());
        // a multi-call span cannot attribute verbatim ids per call; render the parsed shape
        return call.name() + "(" + JsonCodec.stringify(call.arguments()) + ")";
    }
}
