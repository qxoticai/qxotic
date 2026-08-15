package com.qxotic.jinfer.x.examples.inflect2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AudioIOTest {

    @Test
    void samplesConvertAndClamp() {
        // full scale, silence, half scale, and values past the ends
        byte[] bytes = AudioIO.toS16LE(new float[] {1f, 0f, -0.5f, 2f, -2f});
        ByteBuffer samples = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(32767, samples.getShort());
        assertEquals(0, samples.getShort());
        assertEquals(-16383, samples.getShort());
        assertEquals(32767, samples.getShort(), "above full scale clamps");
        assertEquals(-32768, samples.getShort(), "below full scale clamps");
    }

    @Test
    void writesAReadableWavHeader(@TempDir Path directory) throws IOException {
        Path file = directory.resolve("tone.wav");
        float[] pcm = new float[100];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) Math.sin(i / 5.0) * 0.5f;
        AudioIO.writeWav(pcm, 24000, file);

        byte[] written = Files.readAllBytes(file);
        assertEquals(44 + pcm.length * 2, written.length, "44-byte header plus 16-bit samples");
        ByteBuffer header = ByteBuffer.wrap(written).order(ByteOrder.LITTLE_ENDIAN);
        byte[] tag = new byte[4];
        header.get(tag);
        assertEquals("RIFF", new String(tag, StandardCharsets.US_ASCII));
        assertEquals(written.length - 8, header.getInt());
        header.get(tag);
        assertEquals("WAVE", new String(tag, StandardCharsets.US_ASCII));
        header.get(tag);
        assertEquals("fmt ", new String(tag, StandardCharsets.US_ASCII));
        assertEquals(16, header.getInt(), "fmt chunk size");
        assertEquals(1, header.getShort(), "PCM");
        assertEquals(1, header.getShort(), "mono");
        assertEquals(24000, header.getInt(), "sample rate");
        assertEquals(24000 * 2, header.getInt(), "byte rate");
        assertEquals(2, header.getShort(), "block align");
        assertEquals(16, header.getShort(), "bits per sample");
        header.get(tag);
        assertEquals("data", new String(tag, StandardCharsets.US_ASCII));
        assertEquals(pcm.length * 2, header.getInt(), "data size");

        // the samples themselves survive the round trip
        byte[] expected = AudioIO.toS16LE(pcm);
        assertArrayEquals(expected, Arrays.copyOfRange(written, 44, written.length));
    }
}
