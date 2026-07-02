// javax.sound.sampled audio decoder: decodes WAV/AIFF/AU natively (no external process) and converts to
// 16 kHz mono float PCM via the JDK's built-in format converters. JVM-only - AudioSystem's ServiceLoader/
// SPI discovery is fragile under GraalVM native-image, so AudioCodec loads this reflectively and never
// pulls it into a native image. For anything AudioSystem can't handle (mp3/flac/ogg without SPI providers,
// or an unsupported target conversion) it falls back to ffmpeg, so the facade always yields 16 kHz mono.
package com.qxotic.jinfer;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;

public final class JavaSoundAudioDecoder implements AudioDecoder {

    private static final int SR = FfmpegAudioDecoder.SAMPLE_RATE;   // 16000
    private final FfmpegAudioDecoder fallback = new FfmpegAudioDecoder();

    @Override
    public String name() {
        return "javasound";
    }

    @Override
    public Media.Audio load(Path path) throws IOException {
        try (AudioInputStream in = AudioSystem.getAudioInputStream(path.toFile())) {
            return convert(in);
        } catch (UnsupportedAudioFileException | IllegalArgumentException e) {
            return fallback.load(path);   // mp3/compressed, or a conversion AudioSystem can't do -> ffmpeg
        }
    }

    @Override
    public Media.Audio decode(byte[] encoded) throws IOException {
        try (AudioInputStream in = AudioSystem.getAudioInputStream(
                new BufferedInputStream(new ByteArrayInputStream(encoded)))) {
            return convert(in);
        } catch (UnsupportedAudioFileException | IllegalArgumentException e) {
            return fallback.decode(encoded);
        }
    }

    /** Convert any decoded stream to 16 kHz mono float[] PCM via the JDK's format converter. Throws
     *  {@link IllegalArgumentException} if the target conversion is unsupported (caller falls back). */
    private Media.Audio convert(AudioInputStream in) throws IOException {
        // Target: signed 16-bit little-endian PCM, 16 kHz, mono. The converter downmixes + resamples.
        AudioFormat target = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, SR, 16, 1, 2, SR, false);
        byte[] bytes;
        try (AudioInputStream conv = AudioSystem.getAudioInputStream(target, in)) {
            bytes = conv.readAllBytes();
        }
        int n = bytes.length / 2;
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) {
            short s = (short) ((bytes[2 * i] & 0xff) | (bytes[2 * i + 1] << 8));   // little-endian
            pcm[i] = s / 32768f;
        }
        return new Media.Audio(pcm, SR, 1);
    }
}
