package com.qxotic.toknroll;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.advanced.SymbolCodec;
import com.qxotic.toknroll.advanced.TokenizationPipeline;
import com.qxotic.toknroll.impl.BpeMergeTable;
import com.qxotic.toknroll.impl.BpeSymbolEncoder;
import com.qxotic.toknroll.impl.ClassicBPE;
import com.qxotic.toknroll.impl.FastTikToken;
import com.qxotic.toknroll.impl.GenericBPE;
import java.util.Map;
import java.util.Objects;

/** High-level factory methods for building {@link Tokenizer} instances. */
public final class Tokenizers {
    private Tokenizers() {}

    /** Creates a builder for advanced tokenizer pipeline composition. */
    public static TokenizationPipeline.Builder pipeline() {
        return TokenizationPipeline.builder();
    }

    /** Creates a builder for advanced tokenizer pipeline composition around a base tokenizer. */
    public static TokenizationPipeline.Builder pipeline(Tokenizer baseTokenizer) {
        return TokenizationPipeline.builder(Objects.requireNonNull(baseTokenizer, "baseTokenizer"));
    }

    /**
     * Creates the native/classic GPT-2 style BPE tokenizer from TikToken data.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param normalizer text normalizer
     * @param splitter text splitter
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return ClassicBPE.classicFromTiktoken(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                Objects.requireNonNull(specialTokens, "specialTokens"),
                Objects.requireNonNull(normalizer, "normalizer"),
                Objects.requireNonNull(splitter, "splitter"));
    }

    /**
     * Creates the native/classic GPT-2 style BPE tokenizer from TikToken data.
     *
     * <p>Uses {@link Normalizer#identity()}.
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return classicBpe(mergeableRanks, specialTokens, Normalizer.identity(), splitter);
    }

    /**
     * Creates the optimized flat-array TikToken-compatible BPE tokenizer from TikToken data.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param normalizer text normalizer
     * @param splitter text splitter
     */
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return FastTikToken.fromTiktoken(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                Objects.requireNonNull(specialTokens, "specialTokens"),
                Objects.requireNonNull(normalizer, "normalizer"),
                Objects.requireNonNull(splitter, "splitter"));
    }

    /**
     * Creates the optimized flat-array TikToken-compatible BPE tokenizer from TikToken data.
     *
     * <p>Uses {@link Normalizer#identity()}.
     */
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return fastBpe(mergeableRanks, specialTokens, Normalizer.identity(), splitter);
    }

    /**
     * Creates the optimized flat-array TikToken-compatible BPE tokenizer from TikToken data.
     *
     * <p>Uses {@link Normalizer#identity()}.
     */
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return fastBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a generic/reusable BPE tokenizer from TikToken-style merge ranks.
     *
     * <p>Unlike {@link #fastBpe(Map, Map, Normalizer, Splitter)}, this implementation is optimized
     * for flexibility and pluggable symbol codecs.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return GenericBPE.fromTiktoken(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                Objects.requireNonNull(specialTokens, "specialTokens"),
                Objects.requireNonNull(normalizer, "normalizer"),
                Objects.requireNonNull(splitter, "splitter"));
    }

    /**
     * Creates a generic/reusable BPE tokenizer from TikToken-style merge ranks.
     *
     * <p>Uses {@link Normalizer#identity()}.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return genericBpe(mergeableRanks, specialTokens, Normalizer.identity(), splitter);
    }

    /**
     * Creates a generic/reusable BPE tokenizer from TikToken-style merge ranks.
     *
     * <p>Uses {@link Normalizer#identity()} and a regex splitter.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return genericBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /** Creates a generic/reusable BPE tokenizer with an explicit symbol codec. */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter,
            SymbolCodec symbolCodec) {
        return GenericBPE.fromTiktoken(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                Objects.requireNonNull(specialTokens, "specialTokens"),
                Objects.requireNonNull(normalizer, "normalizer"),
                Objects.requireNonNull(splitter, "splitter"),
                Objects.requireNonNull(symbolCodec, "symbolCodec"));
    }

    /** Creates a generic/reusable BPE tokenizer from explicit components. */
    public static Tokenizer genericBpe(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            BpeMergeTable mergeTable,
            BpeSymbolEncoder symbolEncoder) {
        return GenericBPE.create(
                Objects.requireNonNull(vocabulary, "vocabulary"),
                Objects.requireNonNull(normalizer, "normalizer"),
                Objects.requireNonNull(splitter, "splitter"),
                Objects.requireNonNull(mergeTable, "mergeTable"),
                Objects.requireNonNull(symbolEncoder, "symbolEncoder"));
    }

    /**
     * Creates the native/classic GPT-2 style BPE tokenizer from TikToken data.
     *
     * <p>Uses {@link Normalizer#identity()}.
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return classicBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Wraps a tokenizer with an explicit text transform applied before {@link Tokenizer#encode}.
     *
     * <p>This is the opt-in path for potentially lossy input mutation such as Unicode normalization
     * or case folding. Such mutation can break round-trip integrity and is generally discouraged
     * unless explicitly desired.
     */
    public static Tokenizer withTextTransform(Tokenizer tokenizer, Normalizer transform) {
        return pipeline(Objects.requireNonNull(tokenizer, "tokenizer"))
                .normalizer(Objects.requireNonNull(transform, "transform"))
                .build();
    }
}
