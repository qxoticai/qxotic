package com.qxotic.tokenizers.advanced;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;

/**
 * A functional interface for decoding operations that convert token sequences back to readable
 * text. Decoder reverses the transformations applied by splitting and normalization, converting
 * token IDs back to their string representations and handling special encoding schemes.
 *
 * <p>Common decoding strategies:
 *
 * <ul>
 *   <li><b>ByteLevel</b>: Reverses byte-level encoding used by GPT-2, mapping bytes back to visible
 *       characters
 *   <li><b>Metaspace</b>: Replaces special whitespace marker (▁) with actual spaces
 * </ul>
 *
 * <p>Example usage:
 *
 * <pre>
 * // ByteLevel decoder for GPT-2 style tokenizers
 * Decoder byteLevel = Decoder.byteLevel();
 * String text = byteLevel.decode(tokens, vocabulary);
 *
 * </pre>
 *
 * @see Splitter
 * @see Tokenizer
 */
@FunctionalInterface
public interface Decoder {

    /**
     * Decodes a sequence of tokens into readable text.
     *
     * @param tokens the token IDs to decode
     * @param vocabulary the vocabulary to look up token strings
     * @return the decoded text
     * @throws IllegalArgumentException if any token ID is not in the vocabulary
     */
    String decode(IntSequence tokens, Vocabulary vocabulary);

    /** A simple decoder that just concatenates token strings. */
    static Decoder canonical() {
        return (tokens, vocab) -> {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < tokens.length(); i++) {
                String token = vocab.token(tokens.intAt(i));
                if (token != null) {
                    sb.append(token);
                }
            }
            return sb.toString();
        };
    }

    /**
     * Creates a ByteLevel decoder that reverses byte-level encoding. This is used by GPT-2 and
     * similar tokenizers.
     *
     * @return a ByteLevel decoder
     */
    static Decoder byteLevel() {
        return fromCodec(SymbolCodec.BYTE_LEVEL);
    }

    /**
     * Creates a decoder that concatenates token strings and then decodes them using the given
     * codec.
     *
     * @param codec the symbol codec to apply after concatenation
     * @return a decoder backed by the given codec
     */
    static Decoder fromCodec(SymbolCodec codec) {
        return (tokens, vocab) -> {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < tokens.length(); i++) {
                String token = vocab.token(tokens.intAt(i));
                if (token != null) {
                    sb.append(token);
                }
            }
            return codec.decodeToText(sb.toString());
        };
    }

    /**
     * Creates a Metaspace decoder that replaces the special whitespace marker with actual spaces.
     *
     * @param replacement the character used to represent spaces (typically '▁' U+2581)
     * @param prependScheme whether to add a space at the beginning
     * @return a Metaspace decoder
     */
    static Decoder metaspace(char replacement, boolean prependScheme) {
        return (tokens, vocab) -> {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < tokens.length(); i++) {
                String token = vocab.token(tokens.intAt(i));
                if (token != null) {
                    if (sb.length() > 0 || prependScheme) {
                        // Replace the metaspace character with an actual space
                        if (token.startsWith(String.valueOf(replacement))) {
                            sb.append(' ');
                            sb.append(token.substring(1));
                        } else {
                            sb.append(token);
                        }
                    } else {
                        sb.append(token);
                    }
                }
            }

            return sb.toString();
        };
    }
}
