package com.qxotic.jinfer.chat;

/**
 * A spoken reply, requested: speech with its transcript interleaved, or speech alone. A request
 * without one is answered in text, and the reply never switches into speech on its own. Only a
 * model that speaks takes one; the speech arrives through {@link ChatEngine.ReplySink#onSpeech} and
 * as the reply's {@link Content.Speech}.
 */
public final class Speech {

    private static final Speech WITH_TRANSCRIPT = new Speech(true, 0), ONLY = new Speech(false, 0);

    private final boolean transcript;
    private final long seed; // the engine's: the codes' draw, resolved when the request is prepared

    private Speech(boolean transcript, long seed) {
        this.transcript = transcript;
        this.seed = seed;
    }

    /** Speech and the words as text: a voice assistant's reply. */
    public static Speech withTranscript() {
        return WITH_TRANSCRIPT;
    }

    /** Speech alone. */
    public static Speech only() {
        return ONLY;
    }

    /** Whether the words come as text too, interleaved with the speech. */
    public boolean transcript() {
        return transcript;
    }

    Speech seeded(long seed) {
        return new Speech(transcript, seed);
    }

    long seed() {
        return seed;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Speech speech && speech.transcript == transcript;
    }

    @Override
    public int hashCode() {
        return Boolean.hashCode(transcript);
    }

    @Override
    public String toString() {
        return transcript ? "Speech.withTranscript()" : "Speech.only()";
    }
}
