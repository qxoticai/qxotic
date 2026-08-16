package com.qxotic.jinfer.boundary.media;

import com.qxotic.jinfer.boundary.Media;
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

    /** Ten minutes of mono speech; the public float representation occupies about 37 MiB. */
    static final int MAX_SAMPLES = SAMPLE_RATE * 60 * 10;

    @Override
    public Media.Audio load(Path path) throws IOException {
        return toAudio(run(path.toString(), null));
    }

    @Override
    public Media.Audio decode(byte[] data) throws IOException {
        return toAudio(run("pipe:0", data));
    }

    private static byte[] run(String input, byte[] data) throws IOException {
        return Ffmpeg.run(
                ffmpegArgs(input), data, Duration.ofMinutes(2), Math.multiplyExact(MAX_SAMPLES, 4));
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
        return new Media.Audio(pcm, SAMPLE_RATE, 1);
    }
}
