package com.qxotic.jinfer.models.gptoss;

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

/**
 * The Harmony reply grammar as a {@link ReplyParser} - the one model family whose reply is not
 * marker spans. The stream alternates headers and bodies: {@code
 * <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...} - the
 * routing key is the channel NAME, which is plain text between trusted specials, so this parser
 * accumulates header text and switches state on the header/body delimiters ({@code <|message|>}
 * opens a body, {@code <|end|>}/{@code <|start|>} close it).
 *
 * <p>Channel routing: {@code analysis} is the reasoning channel; {@code final} and {@code
 * commentary} preamble text are content ({@code <|return|>}/{@code <|call|>} are stop tokens,
 * inert). Native commentary tool-call parsing has no encode-side counterpart on this port (tool
 * requests take the whole-render path), so calls are recovered by the server's string-scan fallback
 * there.
 */
final class HarmonyReplyParser implements ReplyParser {

    private final Tokenizer tokenizer;
    private final int channelId; // <|channel|>
    private final int messageId; // <|message|>
    private final int endId; // <|end|>
    private final int startId; // <|start|>

    private final PendingUtf8 pending = new PendingUtf8();

    private boolean inBody; // false: accumulating header text (channel name, role)
    private final StringBuilder header = new StringBuilder();
    private boolean reasoningBody; // channel of the current/last body
    private Message message;

    private final StringBuilder reasoningText = new StringBuilder();
    private IntSequence.Builder reasoningIds = IntSequence.newBuilder();
    private final StringBuilder contentText = new StringBuilder();
    private IntSequence.Builder contentIds = IntSequence.newBuilder();

    HarmonyReplyParser(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.channelId = SpecialTokens.require(tokenizer, "<|channel|>");
        this.messageId = SpecialTokens.require(tokenizer, "<|message|>");
        this.endId = SpecialTokens.require(tokenizer, "<|end|>");
        this.startId = SpecialTokens.require(tokenizer, "<|start|>");
    }

    @Override
    public String feed(int token) {
        if (message != null) throw new IllegalStateException("parser already finished");
        if (token == messageId) { // header -> body: the channel name decides the routing
            reasoningBody = header.toString().contains("analysis");
            header.setLength(0);
            inBody = true;
            return "";
        }
        if (token == endId || token == startId) { // body -> header
            String flushed = flushPending();
            inBody = false;
            header.setLength(0);
            return flushed;
        }
        if (token == channelId) {
            return ""; // header punctuation
        }
        if (SpecialTokens.isSpecial(tokenizer, token)) {
            return ""; // <|return|>, <|call|>, other scaffold: inert
        }
        if (!inBody) { // header text: role / channel name, never displayed
            header.append(tokenizer.decode(new int[] {token}));
            return "";
        }
        PendingUtf8.Fragment fragment =
                pending.add(tokenizer.decodeBytes(new int[] {token}), token);
        return fragment == null ? "" : emit(fragment.text(), fragment.ids());
    }

    @Override
    public boolean reasoning() {
        return reasoningBody;
    }

    @Override
    public Message finish() {
        if (message == null) {
            flushPending();
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
            message = new Message(Role.ASSISTANT, parts);
        }
        return message;
    }

    private String emit(String fragment, IntSequence ids) {
        if (fragment.isEmpty()) return "";
        if (reasoningBody) {
            reasoningText.append(fragment);
            reasoningIds.addAll(ids);
        } else {
            contentText.append(fragment);
            contentIds.addAll(ids);
        }
        return fragment;
    }

    private String flushPending() {
        PendingUtf8.Fragment fragment = pending.flush();
        return fragment == null ? "" : emit(fragment.text(), fragment.ids());
    }
}
