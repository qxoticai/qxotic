package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The workload the release build's profile is recorded on (`make libtoknroll` with Oracle GraalVM):
 * what toknroll_encode and toknroll_encode_batch run, plain and with specials, on one thread and on
 * all, for tokenizers of every kind the library has a fast path for, on enwik8 prose and on chat
 * text (an added token every 32-96 bytes), in 2 KB documents.
 *
 * <p>Usage: PgoTraining <enwik8> <source>...
 */
public final class PgoTraining {
    private PgoTraining() {}

    public static void main(String[] args) throws Exception {
        String prose =
                new String(
                        Files.readAllBytes(Path.of(args[0])), 0, 12 << 20, StandardCharsets.UTF_8);
        for (int a = 1; a < args.length; a++) {
            Native.Loaded loaded = Native.Loaded.of(Sources.load(args[a]));
            byte[] chat = chat(prose.substring(8 << 20), loaded.specials(), a);
            byte[] text = prose.substring(0, 8 << 20).getBytes(StandardCharsets.UTF_8);
            long ids = 0;
            for (byte[] t : new byte[][] {text, chat}) {
                for (Specials specials : new Specials[] {null, loaded.specials()}) {
                    ids += documents(loaded.tokenizer(), specials, t, false);
                    ids += documents(loaded.tokenizer(), specials, t, true);
                }
            }
            System.out.println(args[a] + ": " + ids + " ids");
        }
    }

    /** Prose with one of the tokenizer's special tokens every 32 to 96 bytes. */
    private static byte[] chat(String prose, Specials specials, int seed) {
        List<String> names = new ArrayList<>(specials.tokens().keySet());
        if (names.isEmpty()) {
            return prose.getBytes(StandardCharsets.UTF_8);
        }
        Random random = new Random(seed);
        StringBuilder chat = new StringBuilder();
        for (int at = 0; at + 96 < prose.length(); ) {
            int step = 32 + random.nextInt(65);
            chat.append(prose, at, at + step).append(names.get(random.nextInt(names.size())));
            at += step;
        }
        return chat.toString().getBytes(StandardCharsets.UTF_8);
    }

    /** {@code text} as 2 KB documents, cut at character boundaries, each encoded on its own. */
    private static long documents(
            Tokenizer tokenizer, Specials specials, byte[] text, boolean all) {
        int count = (text.length + 2047) / 2048;
        long[] ids = new long[count];
        java.util.function.IntConsumer encode =
                i -> {
                    int from = start(text, i * 2048);
                    int to = start(text, Math.min(text.length, (i + 1) * 2048));
                    IntSequence.Builder out = IntSequence.newBuilder(Math.max(16, (to - from) / 3));
                    Native.encodeUtf8(tokenizer, specials, text, from, to, out);
                    ids[i] = out.size();
                };
        if (all) {
            Workers.run(count, encode);
        } else {
            for (int i = 0; i < count; i++) {
                encode.accept(i);
            }
        }
        long total = 0;
        for (long n : ids) {
            total += n;
        }
        return total;
    }

    private static int start(byte[] text, int i) {
        while (i < text.length && (text[i] & 0xC0) == 0x80) {
            i++;
        }
        return i;
    }
}
