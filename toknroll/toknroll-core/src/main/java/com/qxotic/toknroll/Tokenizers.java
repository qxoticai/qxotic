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

/**
 * Factory methods for building {@link Tokenizer} instances from TikToken-style merge ranks.
 *
 * <p>For most use cases, prefer {@link #fastBpe} — it is the fastest TikToken-compatible
 * implementation. Use {@link #classicBpe} only when exact GPT-2 native behaviour is required, and
 * {@link #genericBpe} when a pluggable {@link com.qxotic.toknroll.advanced.SymbolCodec} is needed.
 */
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
     * Creates a {@link TokenizationPipeline} wrapping the optimized flat-array TikToken-compatible
     * BPE tokenizer. Prefer this over {@link #classicBpe} and {@link #genericBpe} for
     * TikToken-compatible encodings.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param normalizer text normalizer applied before splitting
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        FastTikToken.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .normalizer(Objects.requireNonNull(normalizer, "normalizer"))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping the optimized flat-array TikToken-compatible
     * BPE tokenizer.
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
     * Creates a {@link TokenizationPipeline} wrapping the optimized flat-array TikToken-compatible
     * BPE tokenizer.
     *
     * <p>Uses {@link Normalizer#identity()} and a regex splitter.
     */
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return fastBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping the native/classic GPT-2 style BPE tokenizer.
     *
     * @param mergeableRanks mergeable token ranks
     * @param specialTokens special token to id mapping
     * @param normalizer text normalizer applied before splitting
     * @param splitter text splitter applied before BPE encoding
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        ClassicBPE.classicFromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .normalizer(Objects.requireNonNull(normalizer, "normalizer"))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping the native/classic GPT-2 style BPE tokenizer.
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
     * Creates a {@link TokenizationPipeline} wrapping the native/classic GPT-2 style BPE tokenizer.
     *
     * <p>Uses {@link Normalizer#identity()} and a regex splitter.
     */
    public static Tokenizer classicBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return classicBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping a generic/reusable BPE tokenizer from
     * TikToken-style merge ranks.
     *
     * <p>Unlike {@link #fastBpe}, this implementation is optimized for flexibility and pluggable
     * symbol codecs.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        return TokenizationPipeline.builder(
                        GenericBPE.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens")))
                .normalizer(Objects.requireNonNull(normalizer, "normalizer"))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping a generic/reusable BPE tokenizer.
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
     * Creates a {@link TokenizationPipeline} wrapping a generic/reusable BPE tokenizer.
     *
     * <p>Uses {@link Normalizer#identity()} and a regex splitter.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return genericBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping a generic/reusable BPE tokenizer with an
     * explicit symbol codec.
     */
    public static Tokenizer genericBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter,
            SymbolCodec symbolCodec) {
        return TokenizationPipeline.builder(
                        GenericBPE.fromTiktoken(
                                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                                Objects.requireNonNull(specialTokens, "specialTokens"),
                                Objects.requireNonNull(symbolCodec, "symbolCodec")))
                .normalizer(Objects.requireNonNull(normalizer, "normalizer"))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
    }

    /**
     * Creates a {@link TokenizationPipeline} wrapping a generic/reusable BPE tokenizer from
     * explicit components.
     */
    public static Tokenizer genericBpe(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            BpeMergeTable mergeTable,
            BpeSymbolEncoder symbolEncoder) {
        return TokenizationPipeline.builder(
                        GenericBPE.create(
                                Objects.requireNonNull(vocabulary, "vocabulary"),
                                Objects.requireNonNull(mergeTable, "mergeTable"),
                                Objects.requireNonNull(symbolEncoder, "symbolEncoder")))
                .normalizer(Objects.requireNonNull(normalizer, "normalizer"))
                .splitter(Objects.requireNonNull(splitter, "splitter"))
                .build();
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
