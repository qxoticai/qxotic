package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.*;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class GGUFTokenizerModelFactory {
    private static final char METASPACE = '\u2581';

    /**
     * The {@code tokenizer.ggml.pre} names whose word, when it is a token of its own, encodes as
     * that token without merging: llama.cpp's {@code ignore_merges} (llama-vocab.cpp), Hugging
     * Face's {@code ignore_merges} in the same models' tokenizer.json.
     */
    static final Set<String> IGNORE_MERGES =
            Set.of(
                    "minicpm5",
                    "llama3",
                    "llama-v3",
                    "llama-bpe",
                    "falcon3",
                    "falcon-h1",
                    "midm-2.0",
                    "jina-v5-nano",
                    "lfm2",
                    "youtu",
                    "glm4",
                    "glm5",
                    "tekken",
                    "pixtral",
                    "granite-embed-multi-97m");

    private GGUFTokenizerModelFactory() {}

    static TokenizationModel buildTiktokenModel(GGUF gguf) {
        boolean replaceWhitespaceMarker =
                "gemma4".equals(GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.MODEL));
        Vocabulary vocabulary = buildVocabulary(gguf, replaceWhitespaceMarker, true);
        List<MergeRule> merges = buildMerges(gguf, vocabulary, replaceWhitespaceMarker, true);
        String pre = GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.PRE);
        boolean ignoreMerges = pre != null && IGNORE_MERGES.contains(pre);
        return ImplAccessor.createTiktokenModel(vocabulary, merges, ignoreMerges);
    }

    static TokenizationModel buildSentencePieceModel(GGUF gguf) {
        boolean replaceWhitespaceMarker =
                "gemma4".equals(GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.MODEL));
        Vocabulary vocabulary = buildVocabulary(gguf, replaceWhitespaceMarker, false);
        long[][] packed = buildPackedMerges(gguf, vocabulary, replaceWhitespaceMarker);
        if (packed != null) {
            return ImplAccessor.createSentencePieceBpeModel(vocabulary, packed[0], packed[1]);
        }
        float[] scores = gguf.getValueOrDefault(float[].class, GGUFMetadataKeys.SCORES, null);
        if (scores != null && scores.length > 0) {
            return Toknroll.sentencePieceBpeModel(vocabulary, scores);
        }
        throw new IllegalArgumentException(
                "GGUF metadata missing both tokenizer.ggml.merges and tokenizer.ggml.scores");
    }

    private static Vocabulary buildVocabulary(
            GGUF gguf, boolean replaceWhitespaceMarker, boolean canonicalizeByteLevel) {
        String[] tokens = gguf.getValue(String[].class, GGUFMetadataKeys.TOKENS);
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException(
                    "GGUF metadata key missing or empty: " + GGUFMetadataKeys.TOKENS);
        }

        // a spelling names one token, so an id is what toknroll_id says: llama.cpp refuses the file
        // too (it asserts as many spellings as tokens), skipping an empty one, which it renames
        Map<String, Integer> spelled = new HashMap<>(2 * tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].isEmpty()) continue;
            Integer first = spelled.putIfAbsent(tokens[i], i);
            if (first != null) {
                throw new IllegalArgumentException(
                        GGUFMetadataKeys.TOKENS
                                + "["
                                + i
                                + "] spells '"
                                + tokens[i]
                                + "', as token "
                                + first
                                + " does: each token needs a spelling of its own");
            }
        }

        String[] normalizedTokens = new String[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            normalizedTokens[i] =
                    normalizeTokenSurface(
                            tokens[i], replaceWhitespaceMarker, canonicalizeByteLevel);
        }

        int[] tokenTypesRaw =
                gguf.getValueOrDefault(int[].class, GGUFMetadataKeys.TOKEN_TYPE, null);
        int[] tokenTypes = new int[tokens.length];
        Arrays.fill(tokenTypes, StandardTokenType.NORMAL.getId());
        if (tokenTypesRaw != null) {
            for (int i = 0; i < Math.min(tokenTypesRaw.length, tokenTypes.length); i++) {
                tokenTypes[i] = tokenTypesRaw[i];
            }
        }

        return ImplAccessor.createVocabulary(normalizedTokens, tokenTypes);
    }

    private static List<MergeRule> buildMerges(
            GGUF gguf,
            Vocabulary vocabulary,
            boolean replaceWhitespaceMarker,
            boolean canonicalizeByteLevel) {
        String[] mergesRaw = gguf.getValueOrDefault(String[].class, GGUFMetadataKeys.MERGES, null);
        if (mergesRaw == null) {
            throw new IllegalArgumentException(
                    "GGUF metadata key missing: " + GGUFMetadataKeys.MERGES);
        }

        List<MergeRule> merges = new ArrayList<>(mergesRaw.length);
        int denseRank = 0;
        for (String spec : mergesRaw) {
            if (spec == null) {
                continue;
            }
            int space = spec.indexOf(' ');
            if (space < 0) {
                continue;
            }
            String left =
                    normalizeTokenSurface(spec.substring(0, space), replaceWhitespaceMarker, true);
            String right =
                    normalizeTokenSurface(spec.substring(space + 1), replaceWhitespaceMarker, true);
            int leftId = ImplAccessor.getIdOrNegative(vocabulary, left);
            int rightId = ImplAccessor.getIdOrNegative(vocabulary, right);
            int mergedId = ImplAccessor.getIdOrNegative(vocabulary, left + right);
            if (leftId >= 0 && rightId >= 0) {
                merges.add(MergeRule.of(leftId, rightId, denseRank++));
            }
        }
        return merges;
    }

    private static long[][] buildPackedMerges(
            GGUF gguf, Vocabulary vocabulary, boolean replaceWhitespaceMarker) {
        String[] mergesRaw = gguf.getValueOrDefault(String[].class, GGUFMetadataKeys.MERGES, null);
        if (mergesRaw == null || mergesRaw.length == 0) {
            return null;
        }

        long[] keys = new long[mergesRaw.length];
        long[] values = new long[mergesRaw.length];
        int size = 0;
        int rank = 0;
        for (String spec : mergesRaw) {
            if (spec == null) {
                continue;
            }
            int space = spec.indexOf(' ');
            if (space < 0) {
                continue;
            }
            String left =
                    normalizeTokenSurface(spec.substring(0, space), replaceWhitespaceMarker, false);
            String right =
                    normalizeTokenSurface(
                            spec.substring(space + 1), replaceWhitespaceMarker, false);
            int leftId = ImplAccessor.getIdOrNegative(vocabulary, left);
            int rightId = ImplAccessor.getIdOrNegative(vocabulary, right);
            if (leftId < 0 || rightId < 0) {
                continue;
            }
            int mergedId = ImplAccessor.getIdOrNegative(vocabulary, left + right);
            if (mergedId < 0) {
                continue;
            }
            keys[size] = ImplAccessor.pairKey(leftId, rightId);
            values[size] = ImplAccessor.packMerge(rank, mergedId);
            size++;
            rank++;
        }
        if (size == 0) {
            return null;
        }
        if (size < mergesRaw.length) {
            keys = Arrays.copyOf(keys, size);
            values = Arrays.copyOf(values, size);
        }
        return new long[][] {keys, values};
    }

    private static String normalizeTokenSurface(
            String token, boolean replaceWhitespaceMarker, boolean canonicalizeByteLevel) {
        if (token == null) return null;
        String out = replaceWhitespaceMarker ? token.replace(METASPACE, ' ') : token;
        if (!canonicalizeByteLevel) return out;
        if (ByteLevel.isValidEncoding(out)) return out;
        return ByteLevel.encode(out.getBytes(StandardCharsets.UTF_8));
    }
}
