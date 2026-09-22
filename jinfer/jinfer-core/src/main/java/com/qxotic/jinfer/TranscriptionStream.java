package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;

/**
 * Live audio of any length: samples in, transcription out in final pieces. Each {@link #feed}
 * returns the piece that became final, often empty; final pieces never change, and their texts
 * joined in order are the whole transcript. {@link #partial()} is the provisional transcription of
 * the audio after the last final piece.
 *
 * <p>Calls are synchronous: decoding happens inside them, on the caller's thread. A stream is one
 * serial pipeline, used by one caller at a time, like the state it runs on. Token times are offsets
 * from the start of the stream. Once finished or closed, a stream refuses every call but {@link
 * #close()} with an {@link IllegalStateException}.
 */
public interface TranscriptionStream extends AutoCloseable {

    /** The sample rate {@link #feed} expects, in Hz. */
    int sampleRate();

    /**
     * Appends mono {@code [-1, 1]} PCM from {@code pcm[offset, offset+length)} and returns the
     * transcription that became final, often empty.
     */
    Transcription feed(float[] pcm, int offset, int length);

    /** Appends all of {@code pcm}. */
    default Transcription feed(float[] pcm) {
        return feed(pcm, 0, pcm.length);
    }

    /** Appends decoded audio, which must be mono at {@link #sampleRate()}. */
    default Transcription feed(Media.Audio audio) {
        return feed(AudioFormat.requireMono(audio, sampleRate()));
    }

    /**
     * The provisional transcription of the audio after the last final piece. It may still change,
     * each call replaces the previous one, and each costs a decode, so the caller paces it.
     */
    Transcription partial();

    /** The rest of the transcription, final. The stream is then closed. */
    Transcription finish();

    /** Releases the stream without a final piece; idempotent. */
    @Override
    void close();
}
