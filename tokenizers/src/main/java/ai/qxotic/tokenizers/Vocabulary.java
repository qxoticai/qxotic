package ai.qxotic.tokenizers;

import java.util.Map;
import java.util.NoSuchElementException;

/**
 * Represents a vocabulary for text tokenization, providing bidirectional mapping between token IDs
 * and their string representations. The vocabulary serves as a fundamental component in text tokenization
 * systems, particularly for Large Language Models (LLMs).
 *
 * <p>Each token in the vocabulary has both a string representation and a unique numeric identifier (ID).
 * The vocabulary provides methods for converting between these two representations and verifying the
 * existence of specific tokens or IDs.</p>
 *
 * <p>The vocabulary can be iterated over to access all token-to-ID mappings.</p>
 *
 * @see Tokenizer
 */
public interface Vocabulary extends Iterable<Map.Entry<String, Integer>> {

    /**
     * Returns the total number of tokens in this vocabulary.
     *
     * @return the number of unique tokens in the vocabulary
     */
    int size();

    /**
     * Retrieves the string representation of a token given its ID.
     *
     * @param id the numeric identifier of the token
     * @return the string representation of the token
     * @throws NoSuchElementException if the ID is not present in the vocabulary
     */
    String token(int id);

    /**
     * Retrieves the numeric ID for a given token string.
     *
     * @param text the string representation of the token
     * @return the numeric identifier (ID) of the token
     * @throws NoSuchElementException if the text is not present in the vocabulary
     */
    int id(String text);

    /**
     * Checks if a given token ID exists in this vocabulary.
     *
     * @param id the numeric identifier to check
     * @return true if the ID exists in the vocabulary, false otherwise
     */
    boolean contains(int id);

    /**
     * Checks if a given token string exists in this vocabulary.
     *
     * @param text the token string to check
     * @return true if the token exists in the vocabulary, false otherwise
     */
    boolean contains(String text);

    /**
     * Determines if the token with the given ID belongs to a specific token type.
     * The default implementation returns false, indicating no type information is available.
     *
     * @param id   the numeric identifier of the token to check
     * @param type the token type to check against
     * @return true if the token is of the specified type, false otherwise
     * @throws NoSuchElementException if the ID is not present in the vocabulary
     */
    default boolean isTokenOfType(int id, TokenType type) {
        return false;
    }
}