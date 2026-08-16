package com.qxotic.jinfer.boundary.media;

import com.qxotic.jinfer.boundary.Media;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * The audio seam in both directions: {@link #wav}/{@link #pcm16} encode a {@link
 * com.qxotic.jinfer.boundary.Media.Audio}, and {@link #decoder()} selects and caches an {@link
 * AudioDecoder} - ffmpeg under native-image, {@code javax.sound} on a JVM, overridden by {@code
 * -Djinfer.audioDecoder=ffmpeg|javasound}. Decoded output is always 16 kHz mono.
 *
 * <p>{@code javax.sound} is loaded through a NON-CONSTANT class name on purpose (see {@link
 * Codecs#reflect}) - a constant one would let native-image fold the {@code Class.forName} and pull
 * {@code javax.sound.sampled} in.
 */
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
     * - hand-written rather than javax.sound.sampled (which drags java.desktop into a native image
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
        ByteBuffer out =
                ByteBuffer.allocate(WAV_HEADER_BYTES + samples.length)
                        .order(ByteOrder.LITTLE_ENDIAN);
        var ascii = StandardCharsets.US_ASCII;
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
     * The samples alone, 16-bit signed little-endian - a headerless stream, for a player told the
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
        String choice = Codecs.choice("jinfer.audioDecoder", "javasound");
        return switch (choice) {
            case "ffmpeg" -> new FfmpegAudioDecoder();
            case "javasound" ->
                    // explicit type witness - see ImageCodec
                    Codecs.<AudioDecoder>reflect(
                            "com.qxotic.jinfer.boundary.media.JavaSoundAudioDecoder",
                            FfmpegAudioDecoder::new);
            default ->
                    throw new IllegalArgumentException(
                            "unknown -Djinfer.audioDecoder='"
                                    + choice
                                    + "' (expected 'ffmpeg' or 'javasound')");
        };
    }
}
