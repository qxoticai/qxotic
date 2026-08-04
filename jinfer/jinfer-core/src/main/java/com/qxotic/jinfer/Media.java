package com.qxotic.jinfer;

/**
 * The universal decoded-signal formats for non-text input. Each is the canonical raw a decoder
 * produces — codec-free, at native parameters — and is universal over the LLM-input signal domain
 * (RGB-family raster, PCM waveform, sampled frames). Color models, codecs, containers, HDR and
 * variable frame rate are resolved upstream at decode, deliberately not represented here; anything
 * outside this domain (a depth volume, a multispectral cube) would be a new variant, added when a
 * model actually consumes it.
 *
 * <p>A {@code Media} value plays two roles: its {@code .class} is the modality key for {@link
 * MultiModal#embedder}, and the value itself is the payload the returned {@link Embedder} consumes.
 * The model-paired embedder owns every step from here — resample/resize, channel collapse,
 * normalize — so the caller only ever provides the faithfully-decoded signal at its own native
 * parameters.
 */
public sealed interface Media permits Media.Image, Media.Audio, Media.Video {

    /**
     * Decoded sRGB-family raster: HWC interleaved (channels innermost), values in {@code [0,1]}.
     * {@code channels} is a count with conventional meaning — 1 = gray, 3 = RGB, 4 = RGBA — not a
     * color-space tag (CMYK/YUV/Lab decode to RGB upstream). Layout mirrors the field order {@code
     * [H,W,C]}: {@code values[(y*width + x)*channels + c]}, length {@code height*width*channels}.
     */
    record Image(float[] values, int height, int width, int channels) implements Media {}

    /**
     * Decoded PCM: interleaved (channels innermost), samples in {@code [-1,1]} at {@code
     * sampleRate} Hz. {@code pcm[frame*channels + ch]}; frame count is {@code pcm.length /
     * channels} (derived).
     */
    record Audio(float[] pcm, int sampleRate, int channels) implements Media {

        /**
         * The clips end to end. Every clip must share one {@code sampleRate} and {@code channels} -
         * a mismatch is a resampling job, not a concatenation, and silently picking one of the two
         * would play back at the wrong pitch. An empty list is not a waveform.
         */
        public static Audio concat(java.util.List<Audio> clips) {
            if (clips.isEmpty()) throw new IllegalArgumentException("no clips to join");
            Audio first = clips.get(0);
            int total = 0;
            for (Audio clip : clips) {
                if (clip.sampleRate() != first.sampleRate() || clip.channels() != first.channels())
                    throw new IllegalArgumentException(
                            "clips differ: "
                                    + first.sampleRate()
                                    + " Hz/"
                                    + first.channels()
                                    + "ch vs "
                                    + clip.sampleRate()
                                    + " Hz/"
                                    + clip.channels()
                                    + "ch");
                total += clip.pcm().length;
            }
            float[] pcm = new float[total];
            int at = 0;
            for (Audio clip : clips) {
                System.arraycopy(clip.pcm(), 0, pcm, at, clip.pcm().length);
                at += clip.pcm().length;
            }
            return new Audio(pcm, first.sampleRate(), first.channels());
        }
    }

    /**
     * Sampled frames, each carrying its TRUE position in the source. A sampled video is a sequence
     * of (image, timestamp) pairs - never "frames at some rate": reference processors sample a
     * fixed frame count UNIFORMLY across the whole duration (an hour of video = 32 frames ~113s
     * apart), and the interleaved timestamps are what keep sparse sampling temporally grounded for
     * the model. Constant-rate clips are the special case {@code timestamp = i / fps}. Any sampling
     * policy - uniform, scene-cut, caller-curated - is representable; the pairing makes
     * misalignment unconstructible. (No audio track: no consumer reads one; add it back when a
     * model ingests synchronized audio.)
     */
    record Video(java.util.List<Frame> frames) implements Media {

        /** One sampled frame at {@code timestamp} from the source's start. */
        public record Frame(Image image, java.time.Duration timestamp) {}

        public Video {
            frames = java.util.List.copyOf(frames); // immutable: the ascending check holds forever
            for (int i = 1; i < frames.size(); i++) {
                if (frames.get(i).timestamp().compareTo(frames.get(i - 1).timestamp()) < 0)
                    throw new IllegalArgumentException("frame timestamps must ascend");
            }
        }
    }
}
