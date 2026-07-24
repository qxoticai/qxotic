package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.PendingUtf8;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The Harmony reply grammar as a {@link ReplyParser} - the one model family whose reply is not
 * marker spans. The stream alternates headers and bodies: {@code
 * <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...} - the
 * routing key is the channel NAME, which is plain text between trusted specials, so this parser
 * accumulates header text and switches state on the header/body delimiters ({@code <|message|>}
 * opens a body, {@code <|end|>}/{@code <|start|>} close it).
 *
 * <p>Channel routing: {@code analysis} is the reasoning channel; {@code final} and plain {@code
 * commentary} preamble text are content; a header carrying {@code to=functions.{name}} claims its
 * body as a TOOL CALL - nothing streams from it, the JSON payload becomes a {@link Part.ToolCall}
 * (with verbatim ids) at body close. {@code <|call|>} is both a stop token and the call-body
 * terminator; {@code <|constrain|>} and other scaffold specials are inert. A body whose payload
 * does not parse as a JSON object is no call (constrained/trained decoding makes that rare).
 */
final class HarmonyReplyParser implements ReplyParser {

    private static final String CALL_TARGET = "to=functions.";

    private final Tokenizer tokenizer;
    private final int messageId; // <|message|>
    private final int endId; // <|end|>
    private final int startId; // <|start|>
    private final int callId; // <|call|> (absent in test vocabularies)

    private final PendingUtf8 pending = new PendingUtf8();

    private boolean inBody; // false: accumulating header text (channel name, role)
    private final StringBuilder header = new StringBuilder();
    private boolean reasoningBody; // channel of the current/last body
    private Message message;

    private final StringBuilder reasoningText = new StringBuilder();
    private IntSequence.Builder reasoningIds = IntSequence.newBuilder();
    private final StringBuilder contentText = new StringBuilder();
    private IntSequence.Builder contentIds = IntSequence.newBuilder();

    private String callName; // non-null while the open body is a claimed tool call
    private final StringBuilder callText = new StringBuilder();
    private IntSequence.Builder callIds = IntSequence.newBuilder();
    private final List<Part.ToolCall> calls = new ArrayList<>();

    HarmonyReplyParser(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.messageId = SpecialTokens.require(tokenizer, "<|message|>");
        this.endId = SpecialTokens.require(tokenizer, "<|end|>");
        this.startId = SpecialTokens.require(tokenizer, "<|start|>");
        this.callId = SpecialTokens.find(tokenizer, "<|call|>").orElse(-1);
    }

    @Override
    public String feed(int token) {
        if (message != null) throw new IllegalStateException("parser already finished");
        if (token == messageId) { // header -> body: the header decides the routing
            String h = header.toString();
            callName = callTarget(h);
            reasoningBody = callName == null && h.contains("analysis");
            header.setLength(0);
            inBody = true;
            return "";
        }
        if (token == endId || token == startId || token == callId) { // body -> header
            String flushed = route(pending.flush());
            closeCall();
            inBody = false;
            header.setLength(0);
            return flushed;
        }
        if (SpecialTokens.isSpecial(tokenizer, token)) {
            // <|channel|>, <|constrain|>, other scaffold: inert - but a special BETWEEN header
            // words is a boundary ("get_time<|constrain|>json" must not read as "get_timejson")
            if (!inBody) header.append(' ');
            return "";
        }
        if (!inBody) { // header text: role / channel name / call target, never displayed
            header.append(tokenizer.decode(new int[] {token}));
            return "";
        }
        return route(pending.add(tokenizer.decodeBytes(new int[] {token}), token));
    }

    /** The {@code functions.{name}} target a call header addresses, or null for plain bodies. */
    private static String callTarget(String header) {
        int at = header.indexOf(CALL_TARGET);
        if (at < 0) return null;
        int from = at + CALL_TARGET.length();
        int to = from;
        while (to < header.length() && !Character.isWhitespace(header.charAt(to))) to++;
        return to == from ? null : header.substring(from, to);
    }

    /** Closes an open call body: a JSON-object payload becomes the call, anything else is none. */
    private void closeCall() {
        if (callName == null) return;
        try {
            if (JsonCodec.parse(callText.toString()) instanceof Map<?, ?> args) {
                @SuppressWarnings("unchecked")
                Map<String, Object> arguments = (Map<String, Object>) args;
                calls.add(new Part.ToolCall("", callName, arguments, callIds.build()));
            }
        } catch (RuntimeException malformed) {
            // a span that never held a parseable call is no call
        }
        callName = null;
        callText.setLength(0);
        callIds = IntSequence.newBuilder();
    }

    @Override
    public boolean reasoning() {
        return reasoningBody;
    }

    @Override
    public Message finish() {
        if (message == null) {
            route(pending.flush());
            closeCall(); // a call body ended by the un-fed stop token still closes
            List<Part> parts = new ArrayList<>();
            if (!reasoningText.isEmpty()) {
                IntSequence ids = reasoningIds.build();
                parts.add(
                        new Part.Reasoning(
                                List.of(new Part.Text(reasoningText.toString(), ids)), ids));
            }
            if (!contentText.isEmpty()) {
                parts.add(new Part.Text(contentText.toString(), contentIds.build()));
            }
            parts.addAll(calls);
            message = new Message(Role.ASSISTANT, parts);
        }
        return message;
    }

    /** Routes a completed body fragment: claimed call payload (silent) or displayed channel. */
    private String route(PendingUtf8.Fragment fragment) {
        if (fragment == null || fragment.text().isEmpty()) return "";
        if (callName != null) {
            callText.append(fragment.text());
            callIds.addAll(fragment.ids());
            return "";
        }
        if (reasoningBody) {
            reasoningText.append(fragment.text());
            reasoningIds.addAll(fragment.ids());
        } else {
            contentText.append(fragment.text());
            contentIds.addAll(fragment.ids());
        }
        return fragment.text();
    }
}
