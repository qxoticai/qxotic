// Facade over the pluggable AudioDecoder. Selects the backend at runtime and caches it:
//   -Djinfer.audioDecoder=ffmpeg|javasound   (explicit override)
//   default: ffmpeg under GraalVM native-image (where javax.sound.sampled is impractical),
// javasound on a
//   normal JVM (no process spawn for WAV/AIFF; ffmpeg fallback for mp3/compressed).
// ffmpeg is referenced directly (native-image-safe, always present). javasound is loaded
// REFLECTIVELY via a
// non-constant class name so native-image does not fold the Class.forName and pull
// javax.sound.sampled into
// the image; if requested but unavailable (e.g. inside a native image), it falls back to ffmpeg.
// Output is
// always 16 kHz mono float PCM.
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

public final class AudioCodec {

    private AudioCodec() {}

    private static volatile AudioDecoder decoder;

    /** Decode an audio file into 16 kHz mono float PCM via the selected backend. */
    public static Media.Audio load(Path path) throws IOException {
        return decoder().load(path);
    }

    /** Decode encoded audio bytes into 16 kHz mono float PCM via the selected backend. */
    public static Media.Audio decode(byte[] data) throws IOException {
        return decoder().decode(data);
    }

    private static final int WAV_HEADER_BYTES = 44;
    private static final short PCM_FORMAT = 1, BITS_PER_SAMPLE = 16;

    /**
     * Encode to a 16-bit PCM WAV. The ENCODE side is deliberately independent of {@link #decoder()}
     * — hand-written rather than javax.sound.sampled (which drags java.desktop into a native image
     * for the sake of a 44-byte header) and never touching the decoder backend, which defaults to
     * spawning ffmpeg under native-image and would turn a self-contained binary into one that
     * shells out.
     *
     * <p>Mono only, which is what every speech port produces; a multi-channel {@code audio} is
     * rejected rather than silently downmixed.
     */
    public static byte[] wav(Media.Audio audio) {
        byte[] samples = pcm16(audio);
        int byteRate = audio.sampleRate() * audio.channels() * BITS_PER_SAMPLE / 8;
        java.nio.ByteBuffer out =
                java.nio.ByteBuffer.allocate(WAV_HEADER_BYTES + samples.length)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        var ascii = java.nio.charset.StandardCharsets.US_ASCII;
        out.put("RIFF".getBytes(ascii));
        out.putInt(WAV_HEADER_BYTES - 8 + samples.length); // everything after this field
        out.put("WAVEfmt ".getBytes(ascii));
        out.putInt(16); // fmt chunk size
        out.putShort(PCM_FORMAT);
        out.putShort((short) audio.channels());
        out.putInt(audio.sampleRate());
        out.putInt(byteRate);
        out.putShort((short) (audio.channels() * BITS_PER_SAMPLE / 8)); // block align
        out.putShort(BITS_PER_SAMPLE);
        out.put("data".getBytes(ascii));
        out.putInt(samples.length);
        out.put(samples);
        return out.array();
    }

    /**
     * The samples alone, 16-bit signed little-endian — a headerless stream, for a player told the
     * format on its command line, or for chunks after a header has already gone out.
     */
    public static byte[] pcm16(Media.Audio audio) {
        if (audio.channels() != 1)
            throw new IllegalArgumentException("mono only, got " + audio.channels() + " channels");
        float[] pcm = audio.pcm();
        byte[] bytes = new byte[pcm.length * 2];
        for (int i = 0; i < pcm.length; i++) {
            int sample = Math.clamp((int) (pcm[i] * Short.MAX_VALUE), -32768, 32767);
            bytes[i * 2] = (byte) sample;
            bytes[i * 2 + 1] = (byte) (sample >> 8);
        }
        return bytes;
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
            default ->
                    throw new IllegalArgumentException(
                            "unknown -Djinfer.audioDecoder='"
                                    + choice
                                    + "' (expected 'ffmpeg' or 'javasound')");
        };
    }

    /**
     * Instantiate a decoder by name via reflection. Passing the name as an argument (not a literal
     * at the Class.forName site) keeps native-image from constant-folding it, so the
     * javax.sound.sampled backend stays out of native images. Falls back to ffmpeg if the backend
     * can't load.
     */
    private static AudioDecoder loadReflectively(String className) {
        try {
            return (AudioDecoder) Class.forName(className).getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException | LinkageError e) {
            System.err.println(
                    "audio decoder '"
                            + className
                            + "' unavailable ("
                            + e
                            + "); falling back to ffmpeg");
            return new FfmpegAudioDecoder();
        }
    }
}
