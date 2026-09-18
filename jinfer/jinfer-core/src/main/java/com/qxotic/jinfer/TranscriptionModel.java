package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;

/**
 * A speech-to-text model with reusable runtime state: PCM in, a timed transcript out, the way a
 * language model takes tokens and a speech model takes phonemes. Decoding a container format and
 * resampling are the caller's business ({@code jinfer-codecs} produces the expected PCM); the model
 * takes the samples it was trained on, verbatim.
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

    /** One utterance of mono {@code [-1, 1]} PCM at {@link #sampleRate()}, transcribed whole. */
    Transcription transcribe(S state, float[] pcm);

    /**
     * A live utterance over {@code state}: feed audio as it arrives, poll the evolving transcript.
     * The stream owns the state's serial slot until it is finished or closed. Ports that cannot
     * stream keep the default refusal.
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

    /**
     * One decoded utterance. The audio must already be mono at {@link #sampleRate()} - the shape
     * {@code jinfer-codecs} produces - and anything else is refused rather than silently resampled:
     * a rate mismatch does not fail, it degrades recognition, so the conversion stays the caller's
     * explicit decision.
     */
    default Transcription transcribe(S state, Media.Audio audio) {
        if (audio.sampleRate() != sampleRate() || audio.channels() != 1)
            throw new IllegalArgumentException(
                    "audio is "
                            + audio.sampleRate()
                            + " Hz x"
                            + audio.channels()
                            + " but this model expects mono "
                            + sampleRate()
                            + " Hz - decode through jinfer-codecs' AudioCodec, or resample first");
        return transcribe(state, audio.pcm());
    }

    /** As {@link #transcribe(RuntimeState, Media.Audio)} on a fresh state. */
    default Transcription transcribe(Media.Audio audio) {
        try (S state = newState()) {
            return transcribe(state, audio);
        }
    }
}
