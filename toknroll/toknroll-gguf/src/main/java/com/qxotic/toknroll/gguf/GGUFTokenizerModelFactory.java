package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class GGUFTokenizerModelFactory {
    private GGUFTokenizerModelFactory() {}

    static TokenizationModel buildTiktokenModel(GGUF gguf) {
        boolean replaceWhitespaceMarker = shouldReplaceWhitespaceMarker(gguf);
        Vocabulary vocabulary = buildVocabulary(gguf, replaceWhitespaceMarker, true);
        List<Tokenizers.MergeRule> merges =
                buildMerges(gguf, vocabulary, true, replaceWhitespaceMarker, true);
        return Tokenizers.tiktokenModel(vocabulary, merges);
    }

    static TokenizationModel buildSentencePieceModel(GGUF gguf) {
        boolean replaceWhitespaceMarker = shouldReplaceWhitespaceMarker(gguf);
        Vocabulary vocabulary = buildVocabulary(gguf, replaceWhitespaceMarker, false);
        List<Tokenizers.MergeRule> merges =
                buildMerges(gguf, vocabulary, false, replaceWhitespaceMarker, false);
        if (!merges.isEmpty()) {
            return Tokenizers.sentencePieceBpeModel(vocabulary, merges);
        }
        float[] scores = gguf.getValueOrDefault(float[].class, GGUFMetadataKeys.SCORES, null);
        if (scores != null && scores.length > 0) {
            return Tokenizers.sentencePieceBpeModel(vocabulary, scores);
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

    private static List<Tokenizers.MergeRule> buildMerges(
            GGUF gguf,
            Vocabulary vocabulary,
            boolean required,
            boolean replaceWhitespaceMarker,
            boolean canonicalizeByteLevel) {
        String[] mergesRaw = gguf.getValueOrDefault(String[].class, GGUFMetadataKeys.MERGES, null);
        if (mergesRaw == null) {
            if (required) {
                throw new IllegalArgumentException(
                        "GGUF metadata key missing: " + GGUFMetadataKeys.MERGES);
            }
            return List.of();
        }

        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < vocabulary.size(); i++) {
            String token = vocabulary.token(i);
            if (token != null) {
                tokenToId.put(token, i);
            }
        }

        List<Tokenizers.MergeRule> merges = new ArrayList<>();
        int denseRank = 0;
        for (String spec : mergesRaw) {
            if (spec == null) {
                continue;
            }
            String[] parts = spec.split(" ");
            if (parts.length != 2) {
                continue;
            }
            String left =
                    normalizeTokenSurface(parts[0], replaceWhitespaceMarker, canonicalizeByteLevel);
            String right =
                    normalizeTokenSurface(parts[1], replaceWhitespaceMarker, canonicalizeByteLevel);
            Integer leftId = tokenToId.get(left);
            Integer rightId = tokenToId.get(right);
            Integer mergedId = tokenToId.get(left + right);
            boolean hasMergedSurface = mergedId != null;
            if (leftId != null && rightId != null && (canonicalizeByteLevel || hasMergedSurface)) {
                merges.add(new Tokenizers.MergeRule(leftId, rightId, denseRank++));
            }
        }
        return merges;
    }

    private static String normalizeTokenSurface(
            String token, boolean replaceWhitespaceMarker, boolean canonicalizeByteLevel) {
        if (token == null) {
            return null;
        }
        String out = token;
        if (replaceWhitespaceMarker) {
            out = out.replace('\u2581', ' ');
        }
        if (canonicalizeByteLevel) {
            byte[] bytes =
                    ByteLevel.isValidEncoding(out)
                            ? ByteLevel.decode(out)
                            : out.getBytes(StandardCharsets.UTF_8);
            out = ByteLevel.encode(bytes);
        }
        return out;
    }

    private static boolean shouldReplaceWhitespaceMarker(GGUF gguf) {
        return "gemma4".equals(GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.MODEL));
    }
}
