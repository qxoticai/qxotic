package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.media.Media;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * The spoken replies a later request may refer to by OpenAI's {@code audio.id}: a client sends back
 * {@code {"role": "assistant", "audio": {"id": ...}}}, and the turn replays exactly as the model
 * spoke it. The reply is kept with its record - positions and codes, a few KB - and without its
 * waveform, which replay never reads. Bounded and expiring, like OpenAI's own: ponytail, an
 * in-memory LRU; a persistent store if turns must outlive a restart.
 */
final class SpokenTurns {

    static final Duration LIFETIME = Duration.ofHours(1);
    private static final int CAPACITY = 4096;

    private record Kept(Message turn, long expiresAt) {}

    private final Map<String, Kept> turns =
            new LinkedHashMap<>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, Kept> eldest) {
                    return size() > CAPACITY;
                }
            };

    /** A fresh {@code audio.id}. */
    static String newId() {
        return "audio_" + UUID.randomUUID().toString().replace("-", "");
    }

    /**
     * Keeps the spoken {@code reply} under {@code id}; returns when it expires, in epoch seconds.
     */
    synchronized long put(String id, Message reply) {
        long expiresAt = System.currentTimeMillis() / 1000 + LIFETIME.toSeconds();
        turns.put(id, new Kept(withoutWaveform(reply), expiresAt));
        return expiresAt;
    }

    /** The assistant turn kept under {@code id}. Unknown or expired ids are refused. */
    synchronized Message get(String id) {
        Kept kept = turns.get(id);
        if (kept == null || kept.expiresAt() < System.currentTimeMillis() / 1000) {
            turns.remove(id);
            throw new IllegalArgumentException("audio " + id + " is unknown or expired");
        }
        return kept.turn();
    }

    private static Message withoutWaveform(Message reply) {
        List<Content> content = new ArrayList<>(reply.content().size());
        for (Content part : reply.content()) {
            content.add(
                    part instanceof Content.Media m && m.media() instanceof Media.Audio speech
                            ? new Content.Media(
                                    new Media.Audio(
                                            new float[0], speech.sampleRate(), speech.channels()),
                                    null,
                                    m.caption())
                            : part);
        }
        return new Message(reply.role(), content, reply.verbatim());
    }
}
