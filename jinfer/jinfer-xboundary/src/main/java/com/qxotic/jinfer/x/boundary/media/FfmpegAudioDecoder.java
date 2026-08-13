package com.qxotic.jinfer.x.boundary.media;

import com.qxotic.jinfer.x.boundary.Media;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;

/**
 * The ffmpeg audio decoder: any container or codec to 16 kHz mono float32 in ONE pass - decode,
 * resample and downmix. The resampling is swresample's, not a hand-rolled linear interpolation,
 * which is the reason to shell out rather than convert in Java. Needs ffmpeg on PATH.
 */
public final class FfmpegAudioDecoder implements AudioDecoder {

    /** gemma4ua and every other speech encoder fix the input at 16 kHz mono. */
    public static final int SAMPLE_RATE = 16000;

    @Override
    public Media.Audio load(Path path) throws IOException {
        return toAudio(Ffmpeg.run(ffmpegArgs(path.toString()), null));
    }

    @Override
    public Media.Audio decode(byte[] data) throws IOException {
        return toAudio(Ffmpeg.run(ffmpegArgs("pipe:0"), data));
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
