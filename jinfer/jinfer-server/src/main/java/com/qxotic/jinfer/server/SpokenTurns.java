package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.SpokenTurn;
import com.qxotic.jinfer.media.Media;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * The spoken replies a later request may refer to by OpenAI's {@code audio.id}: a client sends back
 * {@code {"role": "assistant", "audio": {"id": ...}}}, and the turn replays exactly as the model
 * spoke it. Only the record is kept - positions and codes, a few KB - never the waveform. Bounded
 * and expiring, like OpenAI's own: ponytail, an in-memory LRU; a persistent store if turns must
 * outlive a restart.
 */
final class SpokenTurns {

    static final Duration LIFETIME = Duration.ofHours(1);
    private static final int CAPACITY = 4096;

    private record Kept(SpokenTurn turn, String transcript, int rate, long expiresAt) {}

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

    /** Keeps {@code speech} under {@code id}; returns when it expires, in epoch seconds. */
    synchronized long put(String id, Content.Speech speech, String transcript) {
        long expiresAt = System.currentTimeMillis() / 1000 + LIFETIME.toSeconds();
        turns.put(id, new Kept(speech.turn(), transcript, speech.audio().sampleRate(), expiresAt));
        return expiresAt;
    }

    /**
     * The assistant turn kept under {@code id}: its transcript and its spoken record (without the
     * waveform, which replay never reads). Unknown or expired ids are refused.
     */
    synchronized List<Content> get(String id) {
        Kept kept = turns.get(id);
        if (kept == null || kept.expiresAt() < System.currentTimeMillis() / 1000) {
            turns.remove(id);
            throw new IllegalArgumentException("audio " + id + " is unknown or expired");
        }
        return List.of(
                new Content.Text(kept.transcript()),
                new Content.Speech(new Media.Audio(new float[0], kept.rate(), 1), kept.turn()));
    }
}
