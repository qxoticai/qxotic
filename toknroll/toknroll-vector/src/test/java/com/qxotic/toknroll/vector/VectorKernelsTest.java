package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.spi.Kernels;
import java.util.Random;
import org.junit.jupiter.api.Test;

class VectorKernelsTest {

    private static final String[] PIECES = {
        "a",
        "hello world",
        "\n",
        "\t",
        "café",
        "ÿ",
        "Ā",
        "߿",
        "ࠀ",
        "中文",
        "￿",
        "😀",
        "𝐀",
        "\uD800",
        "\uDC00",
        "x\uD800",
        "\uDC00y",
        "\uD800\uD800",
        "\uDC00\uDC00",
        "0123456789abcdefghijklmnopqrstuvwxyz"
    };

    @Test
    void coreFindsTheVectorKernels() {
        assertTrue(Kernels.current() instanceof VectorKernels, Kernels.current().toString());
    }

    @Test
    void convertsLikeTheScalarKernels() {
        Kernels vector = new VectorKernels();
        Kernels scalar = Kernels.scalar();
        Random random = new Random(0x5EC7A11L);
        for (int round = 0; round < 3000; round++) {
            StringBuilder text = new StringBuilder();
            int pieces = random.nextInt(4) == 0 ? 1 + random.nextInt(3000) : 1 + random.nextInt(40);
            for (int i = 0; i < pieces; i++) {
                text.append(PIECES[random.nextInt(PIECES.length)]);
            }
            String s = text.toString();
            int start = random.nextInt(Math.min(8, s.length() + 1));
            int end = Math.max(start, s.length() - random.nextInt(Math.min(8, s.length() + 1)));
            assertArrayEquals(scalar.utf8(s, start, end, 16), vector.utf8(s, start, end, 16), s);
            assertArrayEquals(scalar.utf8(text, start, end, 0), vector.utf8(text, start, end, 0));
        }
    }

    @Test
    void surrogatePairsAtEveryChunkAndLaneEdge() {
        Kernels vector = new VectorKernels();
        Kernels scalar = Kernels.scalar();
        for (int position : new int[] {31, 32, 63, 64, 4094, 4095, 4096, 4097, 8191}) {
            StringBuilder text = new StringBuilder("a".repeat(position));
            text.append("😀").append("b".repeat(70)).append("\uD800").append("c".repeat(70));
            String s = text.toString();
            assertArrayEquals(scalar.utf8(s, 0, s.length(), 8), vector.utf8(s, 0, s.length(), 8));
        }
    }
}
