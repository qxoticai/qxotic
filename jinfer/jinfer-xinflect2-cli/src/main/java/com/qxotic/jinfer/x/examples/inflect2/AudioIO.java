// WAV output for synthesized audio: float [-1,1] mono → 16-bit PCM.
//
// Hand-written rather than javax.sound.sampled, which would pull the java.desktop module into a
// native image for the sake of a 44-byte header.
package com.qxotic.jinfer.x.examples.inflect2;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

final class AudioIO {
    private AudioIO() {}

    private static final int HEADER_BYTES = 44;
    private static final short PCM_FORMAT = 1, MONO = 1, BITS_PER_SAMPLE = 16;

    /** Convert a float [-1,1] waveform to 16-bit signed little-endian samples. */
    static byte[] toS16LE(float[] waveform) {
        byte[] bytes = new byte[waveform.length * 2];
        for (int i = 0; i < waveform.length; i++) {
            int sample = Math.clamp((int) (waveform[i] * Short.MAX_VALUE), -32768, 32767);
            bytes[i * 2] = (byte) sample;
            bytes[i * 2 + 1] = (byte) (sample >> 8);
        }
        return bytes;
    }

    /** Write a float [-1,1] mono waveform as a 16-bit PCM WAV file. */
    static void writeWav(float[] waveform, int sampleRate, Path path) throws IOException {
        int dataBytes = waveform.length * 2;
        int byteRate = sampleRate * MONO * BITS_PER_SAMPLE / 8;
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        header.putInt(HEADER_BYTES - 8 + dataBytes); // size of everything after this field
        header.put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII));
        header.putInt(16); // fmt chunk size
        header.putShort(PCM_FORMAT);
        header.putShort(MONO);
        header.putInt(sampleRate);
        header.putInt(byteRate);
        header.putShort((short) (MONO * BITS_PER_SAMPLE / 8)); // block align
        header.putShort(BITS_PER_SAMPLE);
        header.put("data".getBytes(StandardCharsets.US_ASCII));
        header.putInt(dataBytes);

        try (OutputStream out = Files.newOutputStream(path)) {
            out.write(header.array());
            out.write(toS16LE(waveform));
        }
    }
}
