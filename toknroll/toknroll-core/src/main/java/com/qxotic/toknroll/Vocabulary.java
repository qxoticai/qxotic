package com.qxotic.toknroll;

import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * Bidirectional mapping between token strings and their numeric IDs.
 *
 * <p>{@link #token(int)} and {@link #id(String)} throw on unknown input; use {@link
 * #findToken(int)} and {@link #findId(String)} for optional-returning lookups. Iterating yields all
 * token-to-ID entries.
 *
 * @see Tokenizer
 */
public interface Vocabulary extends Iterable<Map.Entry<String, Integer>> {

    /**
     * Returns the number of tokens in this vocabulary.
     *
     * @return token count
     */
    int size();

    /**
     * Returns the token string for {@code id}.
     *
     * @param id token ID
     * @return token string
     * @throws NoSuchElementException if the ID is not present in the vocabulary
     */
    String token(int id);

    /**
     * Returns the ID for {@code text}.
     *
     * @param text token string
     * @return token ID
     * @throws NoSuchElementException if the text is not present in the vocabulary
     */
    int id(String text);

    /**
     * @param id token ID
     * @return true if the ID exists in the vocabulary
     */
    boolean contains(int id);

    /**
     * @param text token string
     * @return true if the token exists in the vocabulary
     */
    boolean contains(String text);

    /**
     * Looks up the token string for {@code id}.
     *
     * @param id token ID
     * @return the token string, or empty if the ID is not in the vocabulary
     */
    default Optional<String> findToken(int id) {
        return contains(id) ? Optional.of(token(id)) : Optional.empty();
    }

    /**
     * Looks up the ID for {@code text}.
     *
     * @param text token string
     * @return the token ID, or empty if the text is not in the vocabulary
     */
    default OptionalInt findId(String text) {
        return contains(text) ? OptionalInt.of(id(text)) : OptionalInt.empty();
    }

    /**
     * Returns whether the token with the given ID is of {@code type}. The default implementation
     * has no type information and returns {@code false} for any present ID.
     *
     * @param id token ID
     * @param type token type to check against
     * @return true if the token is of the specified type
     * @throws NoSuchElementException if the ID is not present in the vocabulary
     */
    default boolean isTokenOfType(int id, TokenType type) {
        if (!contains(id)) {
            throw new NoSuchElementException("Token id not found: " + id);
        }
        return false;
    }
}
