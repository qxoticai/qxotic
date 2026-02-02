package ai.qxotic.tokenizers.impl;

import ai.qxotic.tokenizers.ByteEncoding;
import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.Vocabulary;
import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.GptBytePairEncodingParams;
import com.knuddels.jtokkit.api.IntArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

class JTokkitAdapter implements Tokenizer {
    private final Vocabulary vocabulary;
    private final Encoding encoding;

    JTokkitAdapter(Vocabulary vocabulary, Encoding encoding) {
        this.vocabulary = vocabulary;
        this.encoding = encoding;
    }

    static Tokenizer create(
            String name,
            Pattern splitPattern,
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens) {
        Map<byte[], Integer> rawMergeableRanks =
                mergeableRanks.entrySet().stream()
                        .collect(
                                Collectors.toMap(
                                        entry -> ByteEncoding.stringToBytes(entry.getKey()),
                                        Map.Entry::getValue));

        GptBytePairEncodingParams params =
                new GptBytePairEncodingParams(name, splitPattern, rawMergeableRanks, specialTokens);

        EncodingRegistry encodingRegistry = Encodings.newLazyEncodingRegistry();
        encodingRegistry.registerGptBytePairEncoding(params);

        Encoding encoding = encodingRegistry.getEncoding(name).orElseThrow();
        Vocabulary vocabulary = createVocabulary(mergeableRanks, specialTokens);

        return new JTokkitAdapter(vocabulary, encoding);
    }

    private static Vocabulary createVocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        int maxMergeableIndex = mergeableRanks.values().stream()
                .mapToInt(Integer::intValue)
                .max()
                .orElse(0);
        int maxSpecialIndex = specialTokens.values().stream()
                .mapToInt(Integer::intValue)
                .max()
                .orElse(0);
        int maxIndex = Math.max(maxMergeableIndex, maxSpecialIndex);

        String[] tokens = new String[maxIndex + 1];
        for (Map.Entry<String, Integer> entry : mergeableRanks.entrySet()) {
            tokens[entry.getValue()] = entry.getKey();
        }
        return VocabularyWithSpecials.create(new VocabularyImpl(tokens), specialTokens);
    }

    @Override
    public Vocabulary vocabulary() {
        return this.vocabulary;
    }

    @Override
    public int countTokens(String text) {
        return encoding.countTokens(text);
    }

    @Override
    public IntSequence encode(String text) {
        return IntSequence.wrap(encoding.encode(text).toArray());
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        IntArrayList intArrayList = new IntArrayList(tokens.length());
        for (int i = 0; i < tokens.length(); i++) {
            intArrayList.add(tokens.intAt(i));
        }
        try {
            return encoding.decodeBytes(intArrayList);
        } catch (IllegalArgumentException e) {
            throw new NoSuchElementException(e);
        }
    }

    @Override
    public String toString() {
        return "Tiktoken (JTokkit): " + encoding.getName();
    }
}
