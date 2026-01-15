package ai.qxotic.tokenizers.impl;

import ai.qxotic.tokenizers.*;

import java.util.List;

/**
 * Abstract base implementation of the {@link Tokenizer} interface that provides a
 * configurable tokenization pipeline with normalization and splitting steps.
 *
 * <p>The tokenization process follows these steps:
 * <ol>
 *   <li>Text normalization using a configured {@link Normalizer}</li>
 *   <li>Text splitting into chunks using a configured {@link TextSplitter}</li>
 *   <li>Token encoding for each chunk using the implementation-specific algorithm</li>
 * </ol>
 *
 * <p>Implementations need to provide:
 * <ul>
 *   <li>The core encoding logic via {@link #encodeImpl(CharSequence)}</li>
 *   <li>The decoding logic via {@link #decodeBytes(IntSequence)}</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * class MyTokenizer extends AbstractTokenizer {
 *     public MyTokenizer(Vocabulary vocab) {
 *         super(vocab, myNormalizer, mySplitter);
 *     }
 *
 *     @Override
 *     protected IntSequence encodeImpl(CharSequence text) {
 *         // Implementation-specific encoding logic
 *     }
 *
 *     @Override
 *     public byte[] decodeBytes(IntSequence tokens) {
 *         // Implementation-specific decoding logic
 *     }
 * }
 * }</pre>
 */
public abstract class AbstractTokenizer implements Tokenizer {
    /**
     * The vocabulary used for token lookup.
     */
    protected final Vocabulary vocabulary;

    /**
     * The normalizer used for text preprocessing.
     */
    protected final Normalizer normalizer;

    /**
     * The splitter used to break text into chunks.
     */
    protected final TextSplitter splitter;

    /**
     * Creates a new tokenizer with custom normalization and splitting behavior.
     *
     * @param vocabulary the vocabulary for token lookup
     * @param normalizer the normalizer for text preprocessing
     * @param splitter   the splitter for breaking text into chunks
     * @throws NullPointerException if any parameter is null
     */
    protected AbstractTokenizer(Vocabulary vocabulary, Normalizer normalizer, TextSplitter splitter) {
        this.vocabulary = vocabulary;
        this.normalizer = normalizer;
        this.splitter = splitter;
    }

    /**
     * Creates a new tokenizer with identity normalization and splitting.
     *
     * @param vocabulary the vocabulary for token lookup
     * @throws NullPointerException if vocabulary is null
     */
    protected AbstractTokenizer(Vocabulary vocabulary) {
        this(vocabulary, Normalizer.IDENTITY, TextSplitter.IDENTITY);
    }

    @Override
    public Vocabulary vocabulary() {
        return this.vocabulary;
    }

    @Override
    public IntSequence encode(String text) {
        IntSequence.Builder tokens = IntSequence.newBuilder();
        CharSequence normalizedPart = normalizer.apply(text);
        List<CharSequence> chunks = splitter.apply(normalizedPart);

        assert CharSequence.compare(
                normalizedPart,
                chunks.stream().reduce(new StringBuilder(), StringBuilder::append, StringBuilder::append)
        ) == 0;

        for (CharSequence chunk : chunks) {
            tokens.addAll(encodeImpl(chunk));
        }
        return tokens.build();
    }

    /**
     * Implements the core encoding logic for a chunk of text. This method is called
     * after normalization and splitting, and should implement the actual token
     * generation algorithm.
     *
     * @param text the chunk of text to encode
     * @return sequence of token IDs representing the text
     * @throws IllegalArgumentException if the text cannot be encoded
     */
    protected abstract IntSequence encodeImpl(CharSequence text);

    @Override
    public abstract byte[] decodeBytes(IntSequence tokens);
}