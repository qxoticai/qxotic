package com.qxotic.jinfer.x.boundary;

import java.time.Duration;
import java.util.List;
import java.util.Objects;

/** Decoded non-text input accepted by an x-native model. */
public sealed interface Media permits Media.Image, Media.Audio, Media.Video {

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

        /** One utterance from clips, in order; every clip must share rate and channel count. */
        public static Audio concat(List<Audio> clips) {
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
    record Video(List<Frame> frames) implements Media {

        /** One sampled frame at {@code timestamp} from the source's start. */
        public record Frame(Image image, Duration timestamp) {}

        public Video {
            frames = List.copyOf(frames); // immutable: the ascending check holds forever
            for (int i = 1; i < frames.size(); i++) {
                if (frames.get(i).timestamp().compareTo(frames.get(i - 1).timestamp()) < 0)
                    throw new IllegalArgumentException("frame timestamps must ascend");
            }
        }
    }
}
