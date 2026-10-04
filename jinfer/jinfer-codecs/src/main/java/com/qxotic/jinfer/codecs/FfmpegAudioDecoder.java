package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

/**
 * The ffmpeg audio decoder: any container or codec to 16 kHz mono float32 in ONE pass - decode,
 * resample and downmix. The resampling is swresample's, not a hand-rolled linear interpolation,
 * which is the reason to shell out rather than convert in Java. Needs ffmpeg on PATH.
 */
public final class FfmpegAudioDecoder implements AudioDecoder {

    /** gemma4ua and every other speech encoder fix the input at 16 kHz mono. */
    public static final int SAMPLE_RATE = 16000;

    /**
     * The longest audio decoded into memory, an hour by default, which the float representation
     * holds in about 220 MiB. {@code -Djinfer.codecs.maxAudioMinutes} raises or lowers it, up to
     * {@link #MAX_MINUTES_CEILING}: ffmpeg's float32 output must fit one byte array.
     */
    static final String MAX_MINUTES = "jinfer.codecs.maxAudioMinutes";

    static final int MAX_MINUTES_CEILING = (Integer.MAX_VALUE - 8) / (4 * SAMPLE_RATE * 60);

    /** The validated minute cap, read per decode so a bad value fails the call, not class init. */
    static int maxMinutes() {
        String value = System.getProperty(MAX_MINUTES);
        if (value == null) return 60;
        int minutes;
        try {
            minutes = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    MAX_MINUTES + " must be an integer, got '" + value + "'");
        }
        if (minutes < 1 || minutes > MAX_MINUTES_CEILING)
            throw new IllegalArgumentException(
                    MAX_MINUTES + " must be in [1, " + MAX_MINUTES_CEILING + "], got " + minutes);
        return minutes;
    }

    /** Fits an int with room for 4 bytes per sample, by the ceiling above. */
    static int maxSamples(int minutes) {
        return SAMPLE_RATE * 60 * minutes;
    }

    static IOException tooLong(int minutes, IOException cause) {
        return new IOException(
                "audio longer than " + minutes + " minutes: raise -D" + MAX_MINUTES, cause);
    }

    @Override
    public String name() {
        return "ffmpeg";
    }

    @Override
    public Media.Audio load(Path path) throws IOException {
        return toAudio(run(path.toString(), null));
    }

    @Override
    public Media.Audio decode(byte[] data) throws IOException {
        return toAudio(run("pipe:0", data));
    }

    private static byte[] run(String input, byte[] data) throws IOException {
        int minutes = maxMinutes();
        try {
            return Subprocess.run(
                    ffmpegArgs(input), data, Duration.ofMinutes(2), maxSamples(minutes) * 4);
        } catch (Subprocess.OutputLimitExceeded e) {
            throw tooLong(minutes, e);
        }
    }

    private static List<String> ffmpegArgs(String input) {
        // -ar 16000 (resample) + -ac 1 (downmix to mono) + -f f32le (raw 32-bit LE float PCM, no
        // int16 quantization, no header). ffmpeg owns decode + resample + downmix in one pass.
        return List.of(
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input,
                "-ar",
                Integer.toString(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "f32le",
                "-");
    }

    private static Media.Audio toAudio(byte[] raw) {
        int n = raw.length / 4; // 4 bytes per float32 sample
        float[] pcm = new float[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(pcm);
        // float decoding overshoots [-1,1] wherever the source clips, so the samples land in
        // range here rather than failing a whole recording over a few loud ones
        for (int i = 0; i < n; i++)
            pcm[i] = Float.isFinite(pcm[i]) ? Math.clamp(pcm[i], -1f, 1f) : 0f;
        return new Media.Audio(pcm, SAMPLE_RATE, 1);
    }
}
