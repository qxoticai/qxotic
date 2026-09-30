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

    /**
     * The pre-tokenization regex of cl100k and Llama 3: {@code (?i:'s|'t|'re|'ve|'m|'ll|'d)|
     * [^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+| \s+(?!\S)|\s+}, a
     * rule set for {@link #split}.
     */
    int CL100K = 0;

    /** {@link #CL100K} with the contraction spelled {@code '[sS]|'[tT]|...}: no long s. */
    int CL100K_ASCII = 1;

    /** {@link #CL100K} with one digit at a time, {@code \p{N}}: Qwen2 and its descendants. */
    int QWEN2 = 2;

    /** {@link #QWEN2} with the contraction spelled {@code '[sS]|'[tT]|...}. */
    int QWEN2_ASCII = 3;

    /**
     * The regex of r50k and GPT-2's byte level pre-tokenizer: {@code 's|'t|'re|'ve|'m|'ll|'d|
     * ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+}.
     */
    int R50K = 4;

    /**
     * The regex of o200k, gpt-4o and gpt-oss: {@code [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}
     * \p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|} the same with {@code
     * +} and {@code *} swapped {@code |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|
     * \s*[\r\n]+|\s+(?!\S)|\s+}.
     */
    int O200K = 5;

    /** {@link #O200K} without the contraction suffix and with one digit at a time: Tekken. */
    int TEKKEN = 6;

    /**
     * Cuts pieces from {@code text[start, end)} under one of the rule sets above, in the order the
     * regex finds them: each entry of {@code ends} the end of one piece, until the array is full or
     * the text ends. Returns how many, 0 when the kernel cannot cut a piece here (the caller then
     * scans by hand), or -1 when it does not implement {@code rules}. {@code start} must be a piece
     * boundary and the text UTF-8; nothing past {@code end} is read.
     */
    int split(int rules, byte[] text, int start, int end, int[] ends);

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
