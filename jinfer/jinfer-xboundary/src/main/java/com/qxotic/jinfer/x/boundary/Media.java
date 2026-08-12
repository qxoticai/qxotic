package com.qxotic.jinfer.x.boundary;

import java.util.Objects;

/** Decoded non-text input accepted by an x-native model. */
public sealed interface Media permits Media.Image, Media.Audio {

    /**
     * Decoded sRGB-family raster in HWC order with values in {@code [0,1]}. The values array is
     * borrowed and must not be mutated while an embedder is using it.
     */
    record Image(float[] values, int height, int width, int channels) implements Media {
        public Image {
            Objects.requireNonNull(values, "values");
            if (height <= 0 || width <= 0)
                throw new IllegalArgumentException("image dimensions must be positive");
            if (channels != 1 && channels != 3 && channels != 4)
                throw new IllegalArgumentException("image channels must be 1, 3, or 4");
            long expected = Math.multiplyExact(Math.multiplyExact((long) height, width), channels);
            if (values.length != expected)
                throw new IllegalArgumentException(
                        "image values length " + values.length + " != " + expected);
            for (float value : values) {
                if (!(value >= 0.0f && value <= 1.0f))
                    throw new IllegalArgumentException("image values must be finite and in [0,1]");
            }
        }
    }

    /**
     * Decoded float PCM, interleaved with channels innermost and samples in {@code [-1,1]}. The PCM
     * array is borrowed and must not be mutated while an embedder is using it.
     */
    record Audio(float[] pcm, int sampleRate, int channels) implements Media {
        public Audio {
            Objects.requireNonNull(pcm, "pcm");
            if (sampleRate <= 0) throw new IllegalArgumentException("sample rate must be positive");
            if (channels <= 0)
                throw new IllegalArgumentException("audio channels must be positive");
            if (pcm.length % channels != 0)
                throw new IllegalArgumentException(
                        "PCM length " + pcm.length + " is not a whole number of frames");
            for (float sample : pcm) {
                if (!(sample >= -1.0f && sample <= 1.0f))
                    throw new IllegalArgumentException("PCM samples must be finite and in [-1,1]");
            }
        }
    }
}
