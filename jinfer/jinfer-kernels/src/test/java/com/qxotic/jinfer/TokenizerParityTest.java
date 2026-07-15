package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Tokenizer ground-truth test: GgufTokenizer (com.qxotic:toknroll with the lfm2 pre-tokenizer) vs
 * token ids captured from {@code llama-tokenize} (llama.cpp) on the LFM2.5-8B GGUF — the reference
 * implementation the model was built against. Also checks round-trip fidelity, that special-token
 * strings embedded in text are NOT mapped by encode, and that decode renders special tokens as
 * their literal text. Requires the model file; skips cleanly when absent.
 */
public final class TokenizerParityTest {

    private static int failures = 0;

    // (text, expected ids) pairs captured via: llama-tokenize -m LFM2.5-8B-A1B-Q8_0.gguf -f <text>
    // --ids
    // (leading BOS stripped; llama.cpp b? June 2026)
    private static final Object[][] EXPECTED = {
        {"Hello, world!", new int[] {35808, 20, 1530, 9}},
        {"  leading and trailing spaces  ", new int[] {229, 5344, 309, 61439, 12413, 266}},
        {"tabs\tand\nnewlines\r\n", new int[] {92, 7453, 206, 404, 207, 3013, 10389, 52099}},
        {
            "unicode: ñé漢字🚀 ½ µ — “quotes” …",
            new int[] {
                105746, 34, 531, 118, 359, 107582, 12494, 90062, 231, 229, 14583, 22775, 1679, 870,
                361, 8841, 737, 6855
            }
        },
        {
            "code: for (int i = 0; i < n; i++) { x += a[i]*b[i]; }",
            new int[] {
                3454, 34, 374, 342, 661, 723, 550, 229, 24, 35, 723, 927, 316, 35, 723, 21099, 695,
                2552, 12388, 267, 10146, 64939, 74, 10146, 28699, 1979
            }
        },
        {
            "numbers 1234567890 3.14159 1e-5 0xDEADBEEF",
            new int[] {
                91846, 229, 9792, 25677, 23944, 24, 229, 27, 22, 13387, 5098, 229, 25, 77, 21, 29,
                229, 24, 96, 4985, 3875, 42, 7881, 46
            }
        },
        {
            "mixedCASE WordsAnd    multiple   spaces",
            new int[] {85, 4452, 116778, 28574, 3982, 360, 4820, 266, 12413}
        },
        {
            "'apostrophes' don't can't won't it's",
            new int[] {11890, 499, 1526, 1501, 15, 1507, 1400, 510, 1400, 4245, 1400, 435, 589}
        },
        {
            "non-breaking space and thin spaces",
            new int[] {7270, 31696, 1437, 718, 6830, 309, 3574, 88493, 3574, 979, 2549}
        },
        {" ", new int[] {229}},
        {"\n", new int[] {207}},
        {"a", new int[] {73}},
        {"", new int[] {}},
    };

    // prompt.txt (with trailing newline) per llama-tokenize: 2121 tokens
    private static final int PROMPT_COUNT = 2121;
    private static final int[] PROMPT_FIRST = {6427, 207, 2603, 2494, 278, 2272, 2803, 34};
    private static final int[] PROMPT_LAST = {2347, 48672, 284, 8};

    public static void main(String[] args) throws Exception {
        Path model =
                Path.of(args.length > 0 ? args[0] : "../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("TokenizerParityTest: model not found (" + model + "), skipping");
            return;
        }
        GGUF gguf;
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            gguf = ModelLoader.readGguf(channel, model.toString());
        }
        GgufTokenizer tokenizer = new GgufTokenizer(gguf);

        for (Object[] testCase : EXPECTED) {
            String text = (String) testCase[0];
            int[] expected = (int[]) testCase[1];
            int[] actual = tokenizer.encode(text).toArray();
            if (!Arrays.equals(expected, actual)) {
                failures++;
                System.err.printf(
                        "ENCODE MISMATCH on %s: expected %s got %s%n",
                        text.length() > 40 ? text.substring(0, 40) + "..." : "\"" + text + "\"",
                        Arrays.toString(expected),
                        Arrays.toString(actual));
            }
            String roundTrip = tokenizer.decode(tokenizer.encode(text));
            if (!text.equals(roundTrip)) {
                failures++;
                System.err.println("ROUND-TRIP MISMATCH on: " + text);
            }
        }

        if (Files.exists(Path.of("prompt.txt"))) {
            List<Integer> ids = tokenizer.encode(Files.readString(Path.of("prompt.txt"))).toList();
            int[] first =
                    ids.subList(0, PROMPT_FIRST.length).stream()
                            .mapToInt(Integer::intValue)
                            .toArray();
            int[] last =
                    ids.subList(ids.size() - PROMPT_LAST.length, ids.size()).stream()
                            .mapToInt(Integer::intValue)
                            .toArray();
            if (ids.size() != PROMPT_COUNT
                    || !Arrays.equals(first, PROMPT_FIRST)
                    || !Arrays.equals(last, PROMPT_LAST)) {
                failures++;
                System.err.printf(
                        "PROMPT MISMATCH: count=%d (expected %d) first=%s last=%s%n",
                        ids.size(), PROMPT_COUNT, Arrays.toString(first), Arrays.toString(last));
            }
        }

        // specials: encode must NOT map them; decode must render them literally
        Map<String, Integer> specials = tokenizer.getSpecialTokens();
        int imStart = specials.get("<|im_start|>");
        String embedded = "<|im_start|>not special here<|im_end|>";
        for (int id : tokenizer.encode(embedded)) {
            if (tokenizer.isSpecialToken(id)) {
                failures++;
                System.err.println("SPECIALS MAPPED BY ENCODE: id " + id);
            }
        }
        if (!"<|im_start|>"
                .equals(new String(tokenizer.decodeTokenBytes(imStart), StandardCharsets.UTF_8))) {
            failures++;
            System.err.println("SPECIAL DECODE MISMATCH for <|im_start|>");
        }
        if (!tokenizer.isSpecialToken(imStart) || tokenizer.isSpecialToken(35808)) {
            failures++;
            System.err.println("isSpecialToken misclassification");
        }

        // encodeWithSpecialTokens (--raw-prompt): specials map to their ids, ordinary text
        // between them encodes exactly as plain encode would
        List<Integer> raw =
                tokenizer.encodeWithSpecialTokens("<|im_start|>Hello, world!<|im_end|>").toList();
        List<Integer> expectedRaw = new ArrayList<>(List.of(imStart));
        expectedRaw.addAll(tokenizer.encode("Hello, world!").toList());
        expectedRaw.add(specials.get("<|im_end|>"));
        if (!raw.equals(expectedRaw)) {
            failures++;
            System.err.println("encodeWithSpecialTokens mismatch: " + raw + " vs " + expectedRaw);
        }

        System.out.println(
                "TokenizerParityTest: vocab="
                        + tokenizer.vocabularySize()
                        + ", failures="
                        + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
