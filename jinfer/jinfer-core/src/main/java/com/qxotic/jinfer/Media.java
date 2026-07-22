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
    record Audio(float[] pcm, int sampleRate, int channels) implements Media {}

    /**
     * Decoded frames at a constant {@code fps}, with an optional synchronized track ({@code audio}
     * may be null for silent video). Variable frame rate and unbounded/streaming sources are out of
     * scope — sample the frames you want before constructing this.
     */
    record Video(Image[] frames, float fps, java.util.Optional<Audio> audio) implements Media {}
}
