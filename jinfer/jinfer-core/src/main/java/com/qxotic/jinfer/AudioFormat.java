package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;

/** The audio format check shared by the transcription APIs. */
final class AudioFormat {

    private AudioFormat() {}

    /**
     * The PCM of {@code audio}, which must be mono at {@code sampleRate}. Anything else is refused
     * rather than resampled: a rate mismatch degrades recognition without failing, so converting
     * stays the caller's explicit decision.
     */
    static float[] requireMono(Media.Audio audio, int sampleRate) {
        if (audio.sampleRate() != sampleRate || audio.channels() != 1)
            throw new IllegalArgumentException(
                    "audio is "
                            + audio.sampleRate()
                            + " Hz x"
                            + audio.channels()
                            + " but mono "
                            + sampleRate
                            + " Hz is expected: decode through jinfer-codecs' AudioCodec, or"
                            + " resample first");
        return audio.pcm();
    }
}
