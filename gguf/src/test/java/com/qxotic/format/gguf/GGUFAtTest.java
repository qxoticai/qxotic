package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** {@link GGUF#at(long)}: a header relocated inside a larger channel. */
public class GGUFAtTest {

    private static GGUF sample() {
        return Builder.newBuilder()
                .putString("general.architecture", "llama")
                .putTensor(TensorEntry.create("w", new long[] {10}, GGMLType.F32, 256))
                .build();
    }

    @Test
    public void tensorDataOffsetShiftsByBase() {
        GGUF gguf = sample();
        GGUF moved = gguf.at(4096);

        assertEquals(gguf.getTensorDataOffset() + 4096, moved.getTensorDataOffset());
        assertEquals(
                gguf.absoluteOffset(gguf.getTensor("w")) + 4096,
                moved.absoluteOffset(moved.getTensor("w")));
    }

    @Test
    public void everythingElseDelegates() {
        GGUF gguf = sample();
        GGUF moved = gguf.at(4096);

        assertEquals(gguf.getVersion(), moved.getVersion());
        assertEquals(gguf.getMetadataKeys(), moved.getMetadataKeys());
        assertEquals("llama", moved.getString("general.architecture"));
        assertEquals(gguf.getType("general.architecture"), moved.getType("general.architecture"));
        assertEquals(gguf.getAlignment(), moved.getAlignment());
        assertSame(gguf.getTensor("w"), moved.getTensor("w"));
        assertEquals(gguf.getTensors(), moved.getTensors());
        assertTrue(moved.containsTensor("w"));
    }

    @Test
    public void zeroBaseIsIdentity() {
        GGUF gguf = sample();
        assertSame(gguf, gguf.at(0));
    }

    @Test
    public void basesCompose() {
        GGUF gguf = sample();
        GGUF twice = gguf.at(100).at(200);

        assertEquals(gguf.getTensorDataOffset() + 300, twice.getTensorDataOffset());
    }

    @Test
    public void negativeBaseIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> sample().at(-1));
    }

    @Test
    public void overflowIsRejected() {
        assertThrows(
                ArithmeticException.class, () -> sample().at(Long.MAX_VALUE).getTensorDataOffset());
    }
}
