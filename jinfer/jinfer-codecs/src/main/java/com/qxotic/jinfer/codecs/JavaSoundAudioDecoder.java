package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * The {@code javax.sound.sampled} decoder: WAV, AIFF and AU with no external process, refusing the
 * rest naming ffmpeg, which reads it. JVM only - {@code AudioSystem}'s SPI discovery is fragile
 * under native-image, so {@link AudioCodec} loads this reflectively.
 */
public final class JavaSoundAudioDecoder implements AudioDecoder {

    private static final int SR = FfmpegAudioDecoder.SAMPLE_RATE; // 16000

    @Override
    public String name() {
        return "javasound";
    }

    @Override
    public Media.Audio load(Path path) throws IOException {
        try (AudioInputStream in = AudioSystem.getAudioInputStream(path.toFile())) {
            return convert(in);
        } catch (UnsupportedAudioFileException | IllegalArgumentException e) {
            throw unreadable(path.toString(), e);
        }
    }

    @Override
    public Media.Audio decode(byte[] encoded) throws IOException {
        try (AudioInputStream in =
                AudioSystem.getAudioInputStream(
                        new BufferedInputStream(new ByteArrayInputStream(encoded)))) {
            return convert(in);
        } catch (UnsupportedAudioFileException | IllegalArgumentException e) {
            throw unreadable("<" + encoded.length + " bytes>", e);
        }
    }

    /** A format, or a conversion to 16 kHz mono, the JDK cannot do: mp3, FLAC, Ogg, m4a. */
    private static IOException unreadable(String src, Exception cause) {
        return new IOException(
                "javax.sound cannot read "
                        + src
                        + ": it reads WAV, AIFF and AU, and ffmpeg reads the rest"
                        + " (install it, or select it with -Djinfer.audioDecoder=ffmpeg)",
                cause);
    }

    /**
     * Convert any decoded stream to 16 kHz mono float[] PCM via the JDK's format converter. Throws
     * {@link IllegalArgumentException} if the target conversion is unsupported.
     */
    private Media.Audio convert(AudioInputStream in) throws IOException {
        // Target: signed 16-bit little-endian PCM, 16 kHz, mono. The converter downmixes +
        // resamples.
        AudioFormat target =
                new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, SR, 16, 1, 2, SR, false);
        byte[] bytes;
        int minutes = FfmpegAudioDecoder.maxMinutes();
        try (AudioInputStream conv = AudioSystem.getAudioInputStream(target, in)) {
            bytes = readBounded(conv, FfmpegAudioDecoder.maxSamples(minutes) * 2);
        } catch (Subprocess.OutputLimitExceeded e) {
            throw FfmpegAudioDecoder.tooLong(minutes, e);
        }
        int n = bytes.length / 2;
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) {
            short s = (short) ((bytes[2 * i] & 0xff) | (bytes[2 * i + 1] << 8)); // little-endian
            pcm[i] = s / 32768f;
        }
        return new Media.Audio(pcm, SR, 1);
    }

    static byte[] readBounded(InputStream input, int maxBytes) throws IOException {
        byte[] bytes = input.readNBytes(maxBytes + 1);
        if (bytes.length > maxBytes)
            throw new Subprocess.OutputLimitExceeded(
                    "decoded audio exceeds the " + maxBytes + "-byte limit");
        return bytes;
    }
}
