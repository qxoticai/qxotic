// WAV file writer for Inflect TTS output (24kHz mono float → 24kHz 16-bit PCM WAV).
package com.qxotic.jinfer.models.inflect2;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

public final class AudioIO {
    private AudioIO() {}

    /** Convert float[-1,1] waveform to 16-bit signed little-endian bytes. */
    public static byte[] toS16LE(float[] waveform) {
        byte[] buf = new byte[waveform.length * 2];
        for (int i = 0; i < waveform.length; i++) {
            int s = (int) (waveform[i] * 32767);
            if (s > 32767) s = 32767;
            else if (s < -32768) s = -32768;
            buf[i * 2] = (byte) s;
            buf[i * 2 + 1] = (byte) (s >> 8);
        }
        return buf;
    }

    /** Write float[-1,1] mono waveform as 24kHz 16-bit WAV. */
    public static void writeWav(float[] waveform, int sampleRate, Path outputPath)
            throws IOException {
        int dataSize = waveform.length * 2;
        try (var out = new FileOutputStream(outputPath.toFile())) {
            ByteBuffer hdr = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
            hdr.put("RIFF".getBytes());
            hdr.putInt(36 + dataSize);
            hdr.put("WAVE".getBytes());
            hdr.put("fmt ".getBytes());
            hdr.putInt(16);
            hdr.putShort((short) 1);
            hdr.putShort((short) 1);
            hdr.putInt(sampleRate);
            hdr.putInt(sampleRate * 2);
            hdr.putShort((short) 2);
            hdr.putShort((short) 16);
            hdr.put("data".getBytes());
            hdr.putInt(dataSize);
            out.write(hdr.array());
            out.write(toS16LE(waveform));
        }
    }
}
