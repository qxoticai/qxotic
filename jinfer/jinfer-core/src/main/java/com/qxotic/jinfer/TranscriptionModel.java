package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;

/**
 * A speech-to-text model with reusable runtime state: PCM in, a timed transcript out. Decoding a
 * container format and resampling are the caller's job ({@code jinfer-codecs} produces the expected
 * PCM).
 *
 * <p>The model is shared; a state is one serial pipeline, used by one call at a time.
 */
public interface TranscriptionModel<C, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Expected input sample rate, in Hz; the PCM handed to {@link #transcribe} must match it. */
    int sampleRate();

    /** Creates state that owns its memory. */
    S newState();

    /**
     * Creates state that borrows caller-owned, cross-thread-accessible memory.
     *
     * <p><b>WARNING: confined arenas MUST NOT be supplied. Misuse can corrupt memory or crash the
     * JVM; a confined arena is refused at load.</b> Even one worker may be a custom pool's thread
     * or a native pthread other than the arena's owner. See {@link Arenas} for the memory contract.
     */
    S newState(MemoryArena<MemorySegment> arena);

    /** Mono {@code [-1, 1]} PCM at {@link #sampleRate()}, of any length, transcribed whole. */
    Transcription transcribe(S state, float[] pcm);

    /**
     * Live audio over {@code state}: feed it as it arrives, poll the evolving transcript. The
     * stream owns the state's serial slot until it is finished or closed. Ports that cannot stream
     * keep the default refusal.
     */
    default TranscriptionStream stream(S state) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support streaming transcription");
    }

    /** Transcribes using a fresh state that is closed before the result is returned. */
    default Transcription transcribe(float[] pcm) {
        try (S state = newState()) {
            return transcribe(state, pcm);
        }
    }

    /** Decoded audio, of any length, which must be mono at {@link #sampleRate()}. */
    default Transcription transcribe(S state, Media.Audio audio) {
        return transcribe(state, AudioFormat.requireMono(audio, sampleRate()));
    }

    /** As {@link #transcribe(RuntimeState, Media.Audio)} on a fresh state. */
    default Transcription transcribe(Media.Audio audio) {
        try (S state = newState()) {
            return transcribe(state, audio);
        }
    }
}
