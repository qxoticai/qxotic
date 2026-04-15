package com.qxotic.toknroll;

import com.qxotic.toknroll.advanced.BpeMergeTable;
import com.qxotic.toknroll.advanced.BpeSymbolEncoder;
import com.qxotic.toknroll.impl.ClassicBPE;
import com.qxotic.toknroll.impl.FastTikToken;
import com.qxotic.toknroll.impl.GenericBPE;
import com.qxotic.toknroll.impl.SymbolCodec;
import java.util.Map;
import java.util.Objects;

/**
 * Factory methods for building {@link Tokenizer} instances from TikToken-style merge ranks.
 *
 * <p>For most use cases, prefer {@link #tikToken} — it is the fastest TikToken-compatible
 * implementation. Use {@link #classicBpe} only when exact GPT-2 native behaviour is required, and
 * {@link #genericBpe} when a pluggable {@link SymbolCodec} is needed.
 */
public final class Tokenizers {
    private Tokenizers() {}

    /**
     * Creates a builder for tokenizer pipeline composition around a base {@link TokenizationModel}.
     */
    public static TokenizationPipeline.Builder pipeline(TokenizationModel model) {
        return TokenizationPipeline.builder(model);
    }

    /**
     * Creates a {@link Tokenizer} wrapping the optimized flat-array TikToken-compatible BPE
     * tokenizer. Prefer this over {@link #classicBpe} and {@link #genericBpe} for
     * TikToken-compatible encodings.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer tikToken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        FastTikToken.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link Tokenizer} wrapping the optimized flat-array TikToken-compatible BPE
     * tokenizer, using a regex splitter.
     */
    public static Tokenizer tikToken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return tikToken(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link Tokenizer} wrapping the native/classic GPT-2 style BPE tokenizer.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        ClassicBPE.classicFromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link Tokenizer} wrapping the native/classic GPT-2 style BPE tokenizer, using a
     * regex splitter.
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return classicBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link Tokenizer} wrapping a generic/reusable BPE tokenizer from TikToken-style
     * merge ranks.
     *
     * <p>Unlike {@link #tikToken}, this implementation is optimized for flexibility and pluggable
     * symbol codecs.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        GenericBPE.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link Tokenizer} wrapping a generic/reusable BPE tokenizer, using a regex
     * splitter.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return genericBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link Tokenizer} wrapping a generic/reusable BPE tokenizer with an explicit symbol
     * codec.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter,
            SymbolCodec symbolCodec) {
        return TokenizationPipeline.builder(
                        GenericBPE.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens"),
                                Objects.requireNonNull(symbolCodec, "symbolCodec")))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link Tokenizer} wrapping a generic/reusable BPE tokenizer from explicit
     * components.
     */
    public static Tokenizer genericBpe(
            Vocabulary vocabulary,
            Splitter splitter,
            BpeMergeTable mergeTable,
            BpeSymbolEncoder symbolEncoder) {
        return TokenizationPipeline.builder(
                        GenericBPE.create(
                                Objects.requireNonNull(vocabulary, "vocabulary"),
                                Objects.requireNonNull(mergeTable, "mergeTable"),
                                Objects.requireNonNull(symbolEncoder, "symbolEncoder")))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * @deprecated use {@link #genericBpe(Vocabulary, Splitter, BpeMergeTable, BpeSymbolEncoder)}.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static Tokenizer genericBpe(
            Vocabulary vocabulary,
            Splitter splitter,
            com.qxotic.toknroll.impl.BpeMergeTable mergeTable,
            com.qxotic.toknroll.impl.BpeSymbolEncoder symbolEncoder) {
        return genericBpe(
                vocabulary, splitter, (BpeMergeTable) mergeTable, (BpeSymbolEncoder) symbolEncoder);
    }
}
