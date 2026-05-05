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

    private ImplAccessor() {}

    public static IntSequence empty() {
        return EMPTY_SEQUENCE;
    }

    public static IntSequence.Builder newBuilder() {
        return new IntSequenceBuilder();
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

    public static Splitter createRegexSplitter(Pattern pattern) {
        return RegexSplitter.create(pattern);
    }

    public static Specials specialsNone() {
        return SpecialsImpl.none();
    }

    public static Specials compileSpecials(Vocabulary vocabulary, Set<String> specials) {
        return SpecialsImpl.compile(vocabulary, specials);
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
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        return TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);
    }

    public static List<MergeRule> reconstructTiktokenMergeRules(
            Map<String, Integer> mergeableRanks) {
        return TiktokenReconstruction.mergeRules(mergeableRanks);
    }

    public static Tokenizer sentencePieceDecodeWrapper(Tokenizer base, boolean trimLeadingSpace) {
        return sentencePieceDecodeWrapper(base, trimLeadingSpace, true);
    }

    public static Tokenizer sentencePieceDecodeWrapper(
            Tokenizer base, boolean trimLeadingSpace, boolean forwardAtStartOfText) {
        return new TransformedTokenizer(base) {
            @Override
            protected String transformDecoded(String decoded, boolean atStartOfText) {
                return TransformedTokenizer.normalizeMetaspaceDecoded(
                        decoded, forwardAtStartOfText && atStartOfText);
            }

            @Override
            protected boolean trimLeadingSpaceAtStart() {
                return trimLeadingSpace;
            }
        };
    }

    public static Tokenizer metaspaceWrapper(Tokenizer base) {
        return new TransformedTokenizer(base) {
            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                String segment = text.subSequence(startInclusive, endExclusive).toString();
                String replaced = segment.replace(' ', '\u2581');
                String prefixed = startInclusive == 0 ? '\u2581' + replaced : replaced;
                base.encodeInto(prefixed, 0, prefixed.length(), out);
            }

            @Override
            protected String transformDecoded(String decoded, boolean atStartOfText) {
                return TransformedTokenizer.normalizeMetaspaceDecoded(decoded, atStartOfText);
            }

            @Override
            protected boolean trimLeadingSpaceAtStart() {
                return true;
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                String segment = text.subSequence(startInclusive, endExclusive).toString();
                String replaced = segment.replace(' ', '\u2581');
                String prefixed = startInclusive == 0 ? '\u2581' + replaced : replaced;
                return base.countTokens(prefixed, 0, prefixed.length());
            }
        };
    }
}
