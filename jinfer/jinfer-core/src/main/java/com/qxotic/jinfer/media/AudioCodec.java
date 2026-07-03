// Facade over the pluggable AudioDecoder. Selects the backend at runtime and caches it:
//   -Djinfer.audioDecoder=ffmpeg|javasound   (explicit override)
//   default: ffmpeg under GraalVM native-image (where javax.sound.sampled is impractical), javasound on a
//   normal JVM (no process spawn for WAV/AIFF; ffmpeg fallback for mp3/compressed).
// ffmpeg is referenced directly (native-image-safe, always present). javasound is loaded REFLECTIVELY via a
// non-constant class name so native-image does not fold the Class.forName and pull javax.sound.sampled into
// the image; if requested but unavailable (e.g. inside a native image), it falls back to ffmpeg. Output is
// always 16 kHz mono float PCM.
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

public final class AudioCodec {

    private AudioCodec() {
    }

    /** gemma4ua and every other speech encoder fix the input at 16 kHz mono. */
    public static final int SAMPLE_RATE = FfmpegAudioDecoder.SAMPLE_RATE;

    private static volatile AudioDecoder decoder;

    /** Decode an audio file into 16 kHz mono float PCM via the selected backend. */
    public static Media.Audio load(Path path) throws IOException {
        return decoder().load(path);
    }

    /** Decode encoded audio bytes into 16 kHz mono float PCM via the selected backend. */
    public static Media.Audio decode(byte[] data) throws IOException {
        return decoder().decode(data);
    }

    /** The active decoder, lazily selected and cached. */
    public static AudioDecoder decoder() {
        AudioDecoder d = decoder;
        if (d == null) {
            synchronized (AudioCodec.class) {
                d = decoder;
                if (d == null) {
                    decoder = d = select();
                }
            }
        }
        return d;
    }

    private static AudioDecoder select() {
        String choice = System.getProperty("jinfer.audioDecoder");
        if (choice == null || choice.isBlank()) {
            boolean nativeImage = System.getProperty("org.graalvm.nativeimage.imagecode") != null;
            choice = nativeImage ? "ffmpeg" : "javasound";
        }
        return switch (choice.toLowerCase(Locale.ROOT)) {
            case "ffmpeg" -> new FfmpegAudioDecoder();
            case "javasound" -> loadReflectively("com.qxotic.jinfer.media.JavaSoundAudioDecoder");
            default -> throw new IllegalArgumentException(
                    "unknown -Djinfer.audioDecoder='" + choice + "' (expected 'ffmpeg' or 'javasound')");
        };
    }

    /** Instantiate a decoder by name via reflection. Passing the name as an argument (not a literal at the
     *  Class.forName site) keeps native-image from constant-folding it, so the javax.sound.sampled backend
     *  stays out of native images. Falls back to ffmpeg if the backend can't load. */
    private static AudioDecoder loadReflectively(String className) {
        try {
            return (AudioDecoder) Class.forName(className).getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException | LinkageError e) {
            System.err.println("audio decoder '" + className + "' unavailable (" + e + "); falling back to ffmpeg");
            return new FfmpegAudioDecoder();
        }
    }
}
