package com.qxotic.jinfer;

/**
 * One live utterance: PCM in as it arrives, the evolving transcript out. Feed any chunk sizes;
 * {@link #partial()} is the current best transcript of everything fed so far and may revise its
 * tail as more audio arrives (text already behind the model's commit horizon no longer changes);
 * {@link #finish()} flushes and returns the final transcript.
 *
 * <p>A stream is one serial pipeline: one caller at a time, like the state it runs on. {@code
 * partial()} computes on demand - the caller paces it - so poll at the cadence your UI needs.
 */
public interface TranscriptionStream extends AutoCloseable {

    /** Appends mono {@code [-1, 1]} PCM at the model's {@link TranscriptionModel#sampleRate()}. */
    void feed(float[] pcm, int offset, int length);

    default void feed(float[] pcm) {
        feed(pcm, 0, pcm.length);
    }

    /** The current best transcript of all audio fed so far; the tail may still be revised. */
    Transcription partial();

    /**
     * The prefix of {@link #partial()} that is behind the commit horizon and can no longer change;
     * empty until something commits. A renderer keeps this text still and repaints only the
     * remainder.
     */
    default Transcription committed() {
        return new Transcription("", java.util.List.of());
    }

    /** Flushes remaining audio and returns the final transcript. The stream is then closed. */
    Transcription finish();

    /** Releases the stream without a final transcript; idempotent. */
    @Override
    void close();
}
