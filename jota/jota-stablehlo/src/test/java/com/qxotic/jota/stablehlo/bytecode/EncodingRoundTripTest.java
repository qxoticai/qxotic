package com.qxotic.jota.stablehlo.bytecode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EncodingRoundTripTest {

    private static final long[] SAMPLES = {
        0,
        1,
        63,
        127,
        128,
        300,
        16383,
        16384,
        1L << 20,
        1L << 35,
        (1L << 56) - 1,
        1L << 56,
        Long.MAX_VALUE,
        -1,
        Long.MIN_VALUE
    };

    @Test
    void varIntsRoundTrip() {
        EncodingWriter w = new EncodingWriter();
        for (long v : SAMPLES) {
            w.writeVarInt(v);
            w.writeSignedVarInt(v);
            w.writeVarIntWithFlag(v >>> 1, (v & 1) != 0);
        }
        EncodingReader r = new EncodingReader(w.toByteArray());
        for (long v : SAMPLES) {
            assertEquals(v, r.readVarInt());
            assertEquals(v, r.readSignedVarInt());
            assertEquals(v >>> 1, r.readVarIntWithFlag());
            assertEquals((v & 1) != 0, r.lastFlag());
        }
        assertTrue(r.isEmpty());
    }

    @Test
    void smallValuesUseOneByte() {
        EncodingWriter w = new EncodingWriter();
        w.writeVarInt(127);
        assertEquals(1, w.size());
        w.writeVarInt(128);
        assertEquals(3, w.size());
        w.writeSignedVarInt(-1); // zigzag: 1
        assertEquals(4, w.size());
    }

    @Test
    void sectionsRoundTrip() {
        EncodingWriter w = new EncodingWriter();
        w.writeSection(4, new byte[] {1, 2, 3});
        EncodingReader r = new EncodingReader(w.toByteArray());
        EncodingReader.Section s = r.readSection();
        assertEquals(4, s.id());
        assertEquals(3, s.data().remaining());
    }
}
