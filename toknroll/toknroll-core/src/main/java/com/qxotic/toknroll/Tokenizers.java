package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ClassicBPE;
import com.qxotic.toknroll.impl.TikTokenModel;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Factory methods for building {@link Tokenizer} instances from TikToken-style merge ranks.
 *
 * <p>For most use cases, prefer {@link #tikToken} — it is the fastest TikToken-compatible
 * implementation. Use {@link #bpe} only when exact GPT-2 native behaviour is required.
 */
public final class Tokenizers {
    private Tokenizers() {}

    /**
     * Creates a {@link Tokenizer} wrapping the optimized flat-array TikToken-compatible BPE
     * tokenizer. Prefer this over {@link #bpe} for TikToken-compatible encodings.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer tikToken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        Splitter checkedSplitter = Objects.requireNonNull(splitter, "splitter");
        return buildTokenizer(
                TikTokenModel.fromTiktoken(
                        Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                        Objects.requireNonNull(specialTokens, "specialTokens")),
                checkedSplitter);
    }

    /**
     * Creates a TikToken-compatible tokenizer from a ranked token table.
     *
     * <p>{@code rankedTokens[tokenId]} provides the token string for each rank/id. All tokens are
     * treated as mergeable.
     *
     * <p>This constructor builds runtime merge behavior from token ranks (ids). It is intended for
     * compiled/token-table formats and does not preserve original merge-list provenance.
     */
    public static Tokenizer tikToken(String[] rankedTokens, Splitter splitter) {
        return tikToken(rankedTokens, null, splitter);
    }

    /**
     * Creates a TikToken-compatible tokenizer from a ranked token table and optional token types.
     *
     * <p>{@code rankedTokens[tokenId]} provides the token string for each rank/id.
     *
     * <p>When {@code tokenTypes} is provided, entries with type {@code 1} are treated as mergeable
     * tokens and all other types are treated as special tokens, except byte-level singleton symbols
     * which are always treated as mergeable.
     *
     * <p>Important: this API reconstructs effective BPE merge priority from token ranks. It is
     * equivalent to rank-table tokenizers, but it does not guarantee reconstruction of an original
     * explicit merges list in non-canonical formats.
     *
     * <p>Special-token representation is caller-controlled. This constructor keeps special token
     * strings exactly as provided and does not apply byte-level encoding/decoding to them.
     */
    public static Tokenizer tikToken(String[] rankedTokens, int[] tokenTypes, Splitter splitter) {
        Objects.requireNonNull(rankedTokens, "rankedTokens");
        if (tokenTypes != null && tokenTypes.length != rankedTokens.length) {
            throw new IllegalArgumentException(
                    "tokenTypes length ("
                            + tokenTypes.length
                            + ") must match rankedTokens length ("
                            + rankedTokens.length
                            + ")");
        }

        Map<String, Integer> mergeableRanks = new LinkedHashMap<>(rankedTokens.length * 2);
        Map<String, Integer> specialTokens = new LinkedHashMap<>();

        for (int tokenId = 0; tokenId < rankedTokens.length; tokenId++) {
            String token = Objects.requireNonNull(rankedTokens[tokenId], "rankedTokens[tokenId]");
            if (tokenTypes == null || tokenTypes[tokenId] == 1 || isByteLevelSingleton(token)) {
                mergeableRanks.put(token, tokenId);
            } else {
                specialTokens.put(token, tokenId);
            }
        }

        return tikToken(mergeableRanks, specialTokens, splitter);
    }

    private static boolean isByteLevelSingleton(String token) {
        try {
            return ByteLevel.decode(token).length == 1;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /**
     * Creates a {@link Tokenizer} wrapping the native GPT-2 style BPE tokenizer.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer bpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        Splitter checkedSplitter = Objects.requireNonNull(splitter, "splitter");
        return buildTokenizer(
                ClassicBPE.classicFromTiktoken(
                        Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                        Objects.requireNonNull(specialTokens, "specialTokens")),
                checkedSplitter);
    }

    private static Tokenizer buildTokenizer(TokenizationModel model, Splitter splitter) {
        return TokenizationPipeline.builder(model)
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }
}
