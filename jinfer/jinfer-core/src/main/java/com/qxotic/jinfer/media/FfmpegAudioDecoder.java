// ffmpeg-based audio decoder: decodes any container/codec (mp3, wav, flac, ogg, m4a, ...) to raw
// PCM,
// resampled to 16 kHz MONO float32 - the universal speech-encoder input. Native-image-safe: no
// javax.sound.sampled (its ServiceLoader/SPI plugin discovery + reflection need build-time config,
// and
// compressed-format support depends on optional providers). ffmpeg owns decode + resample
// (swresample,
// far higher quality than a hand-rolled linear resampler) + downmix in one pass. Requires ffmpeg on
// PATH.
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

public final class FfmpegAudioDecoder implements AudioDecoder {

    /** gemma4ua and every other speech encoder fix the input at 16 kHz mono. */
    public static final int SAMPLE_RATE = 16000;

    @Override
    public String name() {
        return "ffmpeg";
    }

    @Override
    public Media.Audio load(Path path) throws IOException {
        return toAudio(runFfmpeg(ffmpegArgs(path.toString()), null));
    }

    @Override
    public Media.Audio decode(byte[] data) throws IOException {
        return toAudio(runFfmpeg(ffmpegArgs("pipe:0"), data));
    }

    private static List<String> ffmpegArgs(String input) {
        // -ar 16000 (resample) + -ac 1 (downmix to mono) + -f f32le (raw 32-bit LE float PCM, no
        // int16
        // quantization, no header). ffmpeg owns decode + resample + downmix in one pass.
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

    /**
     * Run ffmpeg, streaming any {@code stdin} on a daemon thread and draining stderr concurrently
     * so the child never blocks on a full pipe. Returns the raw stdout bytes (the f32le PCM).
     */
    private static byte[] runFfmpeg(List<String> cmd, byte[] stdin) throws IOException {
        Process p;
        try {
            p = new ProcessBuilder(cmd).start();
        } catch (IOException e) {
            throw new IOException("failed to launch ffmpeg (is it on PATH?): " + e.getMessage(), e);
        }
        if (stdin != null) {
            Thread feeder =
                    new Thread(
                            () -> {
                                try (OutputStream os = p.getOutputStream()) {
                                    os.write(stdin);
                                } catch (IOException ignored) {
                                    // broken pipe if ffmpeg rejects the input and exits early; the
                                    // exit code reports it
                                }
                            },
                            "ffmpeg-stdin");
            feeder.setDaemon(true);
            feeder.start();
        } else {
            p.getOutputStream().close();
        }
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        Thread errDrain =
                new Thread(
                        () -> {
                            try (InputStream es = p.getErrorStream()) {
                                es.transferTo(err);
                            } catch (IOException ignored) {
                            }
                        },
                        "ffmpeg-stderr");
        errDrain.setDaemon(true);
        errDrain.start();

        byte[] out;
        try (InputStream is = p.getInputStream()) {
            out = is.readAllBytes();
        }
        int code;
        try {
            code = p.waitFor();
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while decoding audio", e);
        }
        try {
            errDrain.join(1000);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        if (code != 0) {
            throw new IOException(
                    "ffmpeg exited " + code + ": " + err.toString(StandardCharsets.UTF_8).strip());
        }
        return out;
    }
}
