package com.qxotic.toknroll.spi;

import java.util.ServiceLoader;

/**
 * The hot loops of the encoder that a faster implementation may take over. Found through {@link
 * ServiceLoader}: toknroll-vector provides one written with the Java Vector API. A provider must
 * produce exactly what core's own version does, and is checked against it on a sample before it is
 * used; one that differs, or fails to load, is passed over.
 */
public interface Kernels {

    /**
     * The UTF-8 of {@code text[start, end)} followed by {@code padding} zero bytes. A surrogate
     * without its partner becomes U+FFFD.
     */
    byte[] utf8(CharSequence text, int start, int end, int padding);

    /** Core's own implementation. */
    static Kernels scalar() {
        return ScalarKernels.INSTANCE;
    }

    /**
     * The implementation in use: the first provider that passes the check, else {@link #scalar}.
     */
    static Kernels current() {
        return Loaded.KERNELS;
    }

    /** Loaded once, on first use. */
    final class Loaded {
        static final Kernels KERNELS = load();

        private Loaded() {}

        private static Kernels load() {
            try {
                for (Kernels candidate : ServiceLoader.load(Kernels.class)) {
                    if (agrees(candidate)) {
                        return candidate;
                    }
                }
            } catch (Throwable ignored) {
                // no provider, or one this JVM cannot run (the Vector API module is not enabled)
            }
            return ScalarKernels.INSTANCE;
        }

        private static boolean agrees(Kernels candidate) {
            StringBuilder sample = new StringBuilder("Hello, world! café 中文 ");
            sample.append("😀 x\uD800y \uDC00 \u0085 \n\t");
            for (int i = 0; i < 300; i++) {
                sample.append("abc ").append((char) (0x100 + i)).append(i % 7 == 0 ? "𝐀" : "");
            }
            String text = sample.toString();
            try {
                for (int start : new int[] {0, 1, 7, 64}) {
                    byte[] expected = ScalarKernels.INSTANCE.utf8(text, start, text.length(), 16);
                    byte[] actual = candidate.utf8(text, start, text.length(), 16);
                    if (!java.util.Arrays.equals(expected, actual)) {
                        return false;
                    }
                }
                return true;
            } catch (Throwable e) {
                return false;
            }
        }
    }
}
