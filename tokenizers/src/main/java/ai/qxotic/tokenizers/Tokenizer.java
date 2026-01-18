package ai.qxotic.tokenizers;

import java.nio.charset.StandardCharsets;
import java.util.NoSuchElementException;

/**
 * Provides functionality for converting between text and token sequences using a defined
 * vocabulary. This interface is fundamental to text processing in Large Language Models (LLMs),
 * handling both the encoding of text into token sequences and decoding of token sequences back into
 * text.
 *
 * <p>The tokenizer uses an associated {@link Vocabulary} to perform the conversions and maintains
 * consistency in both directions (encoding and decoding).
 */
public interface Tokenizer {

    /**
     * Returns the vocabulary used by this tokenizer for encoding and decoding operations.
     *
     * @return the vocabulary instance used by this tokenizer
     */
    Vocabulary vocabulary();

    /**
     * Converts text into a sequence of token IDs according to the tokenizer's vocabulary.
     *
     * @param text the input text to encode
     * @return sequence of token IDs representing the input text
     * @throws IllegalArgumentException if the text cannot be encoded
     */
    IntSequence encode(String text);

    /**
     * Counts the number of tokens that would result from encoding the given text. This method may
     * be optimized to be faster than calling {@link #encode(String)} and checking the length.
     *
     * @param text the input text to analyze
     * @return the number of tokens the text would encode to
     */
    default int countTokens(String text) {
        return encode(text).length();
    }

    /**
     * Converts a sequence of token IDs into a UTF-8 string. This is the primary decoding method for
     * most use cases.
     *
     * @param tokens sequence of token IDs to decode
     * @return the decoded text as a String
     * @throws NoSuchElementException if any token ID is not in the vocabulary
     */
    default String decode(IntSequence tokens) {
        return new String(decodeBytes(tokens), StandardCharsets.UTF_8);
    }

    /**
     * Converts a sequence of token IDs into raw bytes. This is a lower-level decoding operation
     * that provides the underlying byte representation.
     *
     * @param tokens sequence of token IDs to decode
     * @return the raw bytes representing the decoded tokens
     * @throws NoSuchElementException if any token ID is not in the vocabulary
     */
    byte[] decodeBytes(IntSequence tokens);
}
