package com.qxotic.toknroll;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.advanced.SymbolCodec;
import com.qxotic.toknroll.advanced.TokenizationModel;
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

    /**
     * Creates a builder for advanced tokenizer pipeline composition around a base {@link
     * TokenizationModel}.
     */
    public static TokenizationPipeline.Builder pipeline(TokenizationModel model) {
        return TokenizationPipeline.builder(model);
    }

    /**
     * Wraps a tokenizer with an explicit text transform applied before {@link Tokenizer#encode}.
     *
     * <p>This is the opt-in path for potentially lossy input mutation such as Unicode normalization
     * or case folding. Such mutation can break round-trip integrity and is generally discouraged
     * unless explicitly desired.
     */
    public static Tokenizer withTextTransform(Tokenizer tokenizer, Normalizer transform) {
        Objects.requireNonNull(transform, "transform");
        return TokenizationPipeline.builder(modelView(tokenizer)).normalizer(transform).build();
    }

    /** Wraps a tokenizer with a pre-encoding splitter stage. */
    public static Tokenizer withSplitter(Tokenizer tokenizer, Splitter splitter) {
        Objects.requireNonNull(splitter, "splitter");
        return TokenizationPipeline.builder(modelView(tokenizer)).splitter(splitter).build();
    }

    private static TokenizationModel modelView(Tokenizer tokenizer) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        if (tokenizer instanceof TokenizationModel) {
            return (TokenizationModel) tokenizer;
        }
        return new TokenizationModel() {
            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                tokenizer.encodeInto(text, startInclusive, endExclusive, out);
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                return tokenizer.countTokens(text, startInclusive, endExclusive);
            }

            @Override
            public int decodeBytesInto(
                    IntSequence tokens, int tokenStartIndex, java.nio.ByteBuffer out) {
                return tokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
            }

            @Override
            public Vocabulary vocabulary() {
                return tokenizer.vocabulary();
            }
        };
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
    public static Tokenizer fastBpe(
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
    public static Tokenizer fastBpe(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            String splitPattern) {
        return fastBpe(mergeableRanks, specialTokens, Splitter.regex(splitPattern));
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
     * <p>Unlike {@link #fastBpe}, this implementation is optimized for flexibility and pluggable
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
}
