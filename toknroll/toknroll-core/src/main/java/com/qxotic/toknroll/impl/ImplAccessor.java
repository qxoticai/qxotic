package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.*;
import java.io.BufferedReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Public bridge to package-private implementation details used by the {@code toknroll-gguf} and
 * {@code toknroll-hf} loader modules.
 */
public final class ImplAccessor {

    private static final IntSequence EMPTY_SEQUENCE = wrap(new int[0]);

    /**
     * Laguna's newline stage as a regex of pieces: a newline run, CRLF included, starts a piece
     * that runs to the next. Followed by Qwen2's regex it takes the staged scanner.
     */
    public static final String LAGUNA_NEWLINES = TiktokenSplitter.LAGUNA_NEWLINES;

    private ImplAccessor() {}

    public static IntSequence empty() {
        return EMPTY_SEQUENCE;
    }

    public static IntSequence.Builder newBuilder() {
        return new IntSequenceBuilder();
    }

    /** Empties a builder from {@link #newBuilder} so that it can be filled again. */
    public static void clear(IntSequence.Builder builder) {
        ((IntSequenceBuilder) builder).clear();
    }

    public static IntSequence.Builder newBuilder(int initialCapacity) {
        return new IntSequenceBuilder(initialCapacity);
    }

    public static IntSequence wrap(int[] array) {
        return new ArrayIntSequence(array);
    }

    public static IntSequence wrap(List<Integer> list) {
        return new ListIntSequence(list);
    }

    public static Vocabulary createVocabularyWithSpecials(
            Map<String, Integer> tokenToId, Map<String, Integer> specialTokens) {
        return VocabularyWithSpecials.create(new VocabularyImpl(tokenToId), specialTokens);
    }

    public static Vocabulary createVocabulary(Map<String, Integer> tokenToId) {
        return new VocabularyImpl(tokenToId);
    }

    /** Creates a vocabulary from token strings and token-type metadata (GGUF path). */
    public static Vocabulary createVocabulary(String[] tokens, int[] tokenTypes) {
        return new VocabularyImpl(tokens, tokenTypes);
    }

    public static TokenizationModel createTiktokenModel(
            Vocabulary vocabulary, List<MergeRule> merges, boolean ignoreMerges) {
        LongLongMap packed =
                merges.isEmpty()
                        ? new LongLongMap(new long[0], new long[0])
                        : TiktokenReconstruction.packTiktokenMerges(vocabulary, merges);
        return TiktokenModel.fromVocabularyAndMerges(vocabulary, packed, ignoreMerges);
    }

    public static TokenizationModel createSentencePieceBpeModel(
            Vocabulary vocabulary, List<MergeRule> merges) {
        LongLongMap packed =
                merges.isEmpty()
                        ? new LongLongMap(new long[0], new long[0])
                        : TiktokenReconstruction.packSentencePieceMerges(vocabulary, merges);
        return SentencePieceBpeModel.fromVocabularyAndMerges(vocabulary, packed);
    }

    public static TokenizationModel createSentencePieceBpeModel(
            Vocabulary vocabulary, long[] mergeKeys, long[] mergeValues) {
        return SentencePieceBpeModel.fromVocabularyAndMerges(
                vocabulary, new LongLongMap(mergeKeys, mergeValues));
    }

    /**
     * Looks up a token ID, returning -1 instead of throwing on miss. Prefer {@link
     * Vocabulary#findId} for new code; this exists for hot loops that need a single HashMap lookup.
     */
    public static int getIdOrNegative(Vocabulary vocabulary, String token) {
        if (vocabulary instanceof VocabularyImpl) {
            return ((VocabularyImpl) vocabulary).getIdOrNegative(token);
        }
        return vocabulary.findId(token).orElse(-1);
    }

    /** Packs a merge rank (high 32 bits) and merged token ID (low 32 bits) into a single long. */
    public static long packMerge(int rank, int mergedId) {
        return SentencePieceBpeModel.packMerge(rank, mergedId);
    }

    public static long pairKey(int leftId, int rightId) {
        return IntPair.of(leftId, rightId);
    }

    public static TokenizationModel createSentencePieceBpeModel(
            Vocabulary vocabulary, float[] scores) {
        return SentencePieceBpeModel.fromVocabulary(vocabulary, scores);
    }

    /**
     * Whether {@code model} behind {@code splitter} is a tiktoken model behind its hand-written
     * scanner, which {@link #encodeScanned} encodes in one pass and an engine may take over.
     */
    public static boolean isScanned(TokenizationModel model, Splitter splitter) {
        return model instanceof TiktokenModel && splitter instanceof TiktokenSplitter;
    }

    /** Encodes {@code text[start, end)} as {@code splitter} cuts it; see {@link #isScanned}. */
    public static void encodeScanned(
            TokenizationModel model,
            Splitter splitter,
            CharSequence text,
            int start,
            int end,
            IntSequence.Builder out) {
        ((TiktokenModel) model).encode(text, start, end, (TiktokenSplitter) splitter, out);
    }

    /** {@link #encodeScanned} for text that is UTF-8 already. */
    public static void encodeScannedUtf8(
            TokenizationModel model,
            Splitter splitter,
            byte[] utf8,
            int start,
            int end,
            IntSequence.Builder out) {
        ((TiktokenModel) model).encodeUtf8(utf8, start, end, (TiktokenSplitter) splitter, out);
    }

    public static Splitter createRegexSplitter(Pattern pattern) {
        return createRegexSplitter(pattern, true);
    }

    /**
     * Every match is a piece; with {@code emitGaps}, so is the text between two matches. A known
     * pattern gets its hand-written scanner; such a pattern matches all text, so it has no gaps.
     */
    public static Splitter createRegexSplitter(Pattern pattern, boolean emitGaps) {
        Splitter scanner = TiktokenSplitter.forPattern(pattern);
        return scanner != null ? scanner : RegexSplitter.create(pattern, emitGaps);
    }

    /**
     * A SentencePiece text in words, cut before a space where no token of {@code tokens} spans the
     * cut, so a merge model gives the whole text's ids. With {@code isolateNewlines}, every newline
     * run is a piece of its own: {@code "ab\n\ncd ef"} is {@code "ab" | "\n\n" | "cd" | " ef"}, and
     * without it {@code "ab\n\ncd" | " ef"}. See {@link SentencePieceSplitter} for the rule.
     */
    public static Splitter createSentencePieceSplitter(String[] tokens, boolean isolateNewlines) {
        return new SentencePieceSplitter(tokens, isolateNewlines);
    }

    /** The scanner for stages it splits alike, or null: see {@link Splitter#sequence}. */
    public static Splitter createStagedSplitter(Splitter[] stages) {
        return TiktokenSplitter.forStages(stages);
    }

    public static Specials specialsNone() {
        return SpecialsImpl.none();
    }

    public static Specials specialsOf(Tokenizer tokenizer) {
        return SpecialsImpl.of(tokenizer);
    }

    public static Map<String, Integer> loadTiktokenMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();
        reader.lines()
                .forEachOrdered(
                        line -> {
                            String[] parts = line.split(" ");
                            byte[] bytes = Base64.getDecoder().decode(parts[0]);
                            String key = ByteLevel.encode(bytes);
                            int value = Integer.parseInt(parts[1]);
                            mergeableRanks.put(key, value);
                        });
        return mergeableRanks;
    }

    public static Vocabulary reconstructTiktokenVocabulary(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Set<String> userDefined) {
        return TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens, userDefined);
    }

    public static List<MergeRule> reconstructTiktokenMergeRules(
            Map<String, Integer> mergeableRanks) {
        return TiktokenReconstruction.mergeRules(mergeableRanks);
    }

    /**
     * A SentencePiece tokenizer as the text sees it, a space ' ' where the vocabulary spells
     * U+2581: encoding runs {@code normalizer} (null for none; the base must not normalize again),
     * then prepends U+2581 as {@code prependScheme} ({@code always}, {@code first} or {@code
     * never}) says; decoding writes each token's bytes, trimming the first token's leading space
     * with {@code trimLeadingSpace}. See {@link MetaspaceTokenizer}.
     */
    public static Tokenizer metaspaceTokenizer(
            Tokenizer base, Normalizer normalizer, String prependScheme, boolean trimLeadingSpace) {
        return new MetaspaceTokenizer(base, normalizer, prependScheme, trimLeadingSpace);
    }
}
