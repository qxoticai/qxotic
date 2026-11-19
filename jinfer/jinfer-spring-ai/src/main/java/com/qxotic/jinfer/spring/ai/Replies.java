package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Verbatim;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import org.springframework.ai.chat.messages.AssistantMessage;

/**
 * The records of this model's recent replies, keyed by what each reply says: its text, reasoning
 * and tool calls. An echo in history that says the same replays the record - the exact tokens the
 * model generated - in place of its re-rendered parts, wherever the echo has been: held in memory,
 * or written to a session or chat memory store and read back. Nothing rides on the message, so a
 * store sees Spring AI's own fields only, and no stored message can carry token ids into a prompt.
 *
 * <p>The reasoning counts stripped: a streamed echo's parts are rebuilt from the deltas, and a
 * reasoning span's seed newline never streamed. The same reply said twice (a bare "OK.") keeps the
 * later record, and replaying it is still the model saying exactly that; tool calls never collide,
 * each carries a fresh id. Thread-safe.
 */
final class Replies {

    // ponytail: 16M token ids (~64 MiB), least recently used first; an evicted echo renders from
    // its parts (one re-prefill, never another answer). A builder knob if a deployment needs more.
    private static final long BUDGET = 1L << 24;

    private final long budget;
    private final LinkedHashMap<ContentKey, Verbatim> recent = new LinkedHashMap<>(16, 0.75f, true);
    private long held; // token ids in recent

    Replies() {
        this(BUDGET);
    }

    Replies(long budget) {
        this.budget = budget;
    }

    /** Keeps {@code record} for echoes of {@code reply}, the message as its caller got it. */
    synchronized void remember(AssistantMessage reply, Verbatim record) {
        if (record.isEmpty()) return;
        Verbatim replaced = recent.put(key(reply), record);
        held += size(record) - (replaced == null ? 0 : size(replaced));
        for (Iterator<Verbatim> eldest = recent.values().iterator();
                held > budget && eldest.hasNext(); ) {
            held -= size(eldest.next());
            eldest.remove();
        }
    }

    /** The record of the reply {@code echo} says, or {@link Verbatim#NONE}. */
    synchronized Verbatim recall(AssistantMessage echo) {
        return recent.getOrDefault(key(echo), Verbatim.NONE);
    }

    private static ContentKey key(AssistantMessage message) {
        List<Object> shown =
                new ArrayList<>(
                        Arrays.asList(
                                message.getText(), JinferMappings.reasoning(message).strip()));
        for (AssistantMessage.ToolCall call : message.getToolCalls())
            shown.add(Arrays.asList(call.id(), call.name(), call.arguments()));
        return ContentKey.sha256(JsonCodec.stringify(shown).getBytes(StandardCharsets.UTF_8));
    }

    private static long size(Verbatim record) {
        return record.positions().length() + record.codes().length();
    }
}
