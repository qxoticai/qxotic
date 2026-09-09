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

    @Test
    public void everyTensorRelocatesByTheSameBase() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("a", new long[] {4}, GGMLType.F32, 0))
                        .putTensor(TensorEntry.create("b", new long[] {4}, GGMLType.F16, 32))
                        .putTensor(TensorEntry.create("c", new long[] {64}, GGMLType.Q8_0, 64))
                        .build();
        GGUF moved = gguf.at(777);

        for (TensorEntry tensor : gguf.getTensors()) {
            assertEquals(gguf.absoluteOffset(tensor) + 777, moved.absoluteOffset(tensor));
        }
    }

    @Test
    public void metadataConveniencesReadThrough() {
        GGUF moved =
                Builder.newBuilder()
                        .putString("s", "v")
                        .putInteger("n", 5)
                        .setAlignment(64)
                        .build()
                        .at(8);

        assertTrue(moved.containsKey("s"));
        assertFalse(moved.containsKey("missing"));
        assertEquals("v", moved.getStringOrDefault("s", "d"));
        assertEquals("d", moved.getStringOrDefault("missing", "d"));
        assertEquals(5, moved.getValue(int.class, "n"));
        assertEquals(9, moved.getValueOrDefault(int.class, "missing", 9));
        assertEquals(MetadataValueType.STRING, moved.getType("s"));
        assertEquals(64, moved.getAlignment());
    }

    @Test
    public void relocatedViewStillDescribesItself() {
        String text = sample().at(64).toString();

        assertNotNull(text);
        assertFalse(text.isBlank());
    }

    @Test
    public void relocationDoesNotTouchTheOriginal() {
        GGUF gguf = sample();
        long before = gguf.getTensorDataOffset();

        gguf.at(4096);

        assertEquals(before, gguf.getTensorDataOffset());
    }
}
