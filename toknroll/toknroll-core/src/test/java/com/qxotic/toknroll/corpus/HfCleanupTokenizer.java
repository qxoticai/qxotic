package com.qxotic.toknroll.corpus;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Wrapper that applies HuggingFace-style {@code clean_up_tokenization_spaces} post-processing to
 * decoded text.
 *
 * <p>This makes the tokenizer output match HF Transformers' default behavior without modifying the
 * core tokenization implementation.
 *
 * <p>Cleanup rules (from transformers/tokenization_utils.py):
 *
 * <ul>
 *   <li>{@code " ?"} → {@code "?"}
 *   <li>{@code " !"} → {@code "!"}
 *   <li>{@code " ."} → {@code "."}
 *   <li>{@code " ,"} → {@code ","}
 *   <li>{@code " ' "} → {@code "'"}
 *   <li>{@code " n't"} → {@code "n't"}
 *   <li>{@code " 're"} → {@code "'re"}
 *   <li>{@code " 've"} → {@code "'ve"}
 *   <li>{@code " 'll"} → {@code "'ll"}
 *   <li>{@code " 'd"} → {@code "'d"}
 *   <li>{@code " 'm"} → {@code "'m"}
 *   <li>{@code " 's"} → {@code "'s"}
 * </ul>
 */
final class HfCleanupTokenizer implements Tokenizer {

    private final Tokenizer delegate;

    HfCleanupTokenizer(Tokenizer delegate) {
        this.delegate = Objects.requireNonNull(delegate);
    }

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        delegate.encodeInto(text, startInclusive, endExclusive, out);
    }

    @Override
    public Vocabulary vocabulary() {
        return delegate.vocabulary();
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        byte[] raw = delegate.decodeBytes(tokens);
        String cleaned = cleanup(new String(raw, StandardCharsets.UTF_8));
        return cleaned.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        // Decode everything then apply cleanup - not ideal for streaming but matches HF behavior
        byte[] raw = delegate.decodeBytes(tokens.subSequence(tokenStartIndex, tokens.length()));
        String cleaned = cleanup(new String(raw, StandardCharsets.UTF_8));
        byte[] cleanedBytes = cleaned.getBytes(StandardCharsets.UTF_8);
        if (cleanedBytes.length > out.remaining()) {
            return -cleanedBytes.length;
        }
        out.put(cleanedBytes);
        return cleanedBytes.length;
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        return delegate.countTokens(text, startInclusive, endExclusive);
    }

    @Override
    public float expectedTokensPerChar() {
        return delegate.expectedTokensPerChar();
    }

    @Override
    public int countBytes(IntSequence tokens) {
        // Count after cleanup to match HF's reported byte count
        return decodeBytes(tokens).length;
    }

    /**
     * Applies HF-style cleanup to decoded text.
     *
     * <p>Order matters: longer patterns must be replaced before shorter ones to avoid
     * double-processing (e.g. " 're" before " ' ").
     */
    static String cleanup(String text) {
        // Use a single pass with StringBuilder for efficiency
        StringBuilder sb = new StringBuilder(text.length());
        int i = 0;
        while (i < text.length()) {
            // Check for patterns at current position
            if (i > 0 && text.charAt(i - 1) == ' ') {
                // Space-prefixed patterns
                if (i + 2 <= text.length()) {
                    char c = text.charAt(i);
                    char c2 = i + 1 < text.length() ? text.charAt(i + 1) : '\0';

                    // Single-char punctuation: " ?", " !", " .", " ,"
                    if (c2 == ' ' || i + 1 == text.length()) {
                        if (c == '?' || c == '!' || c == '.' || c == ',') {
                            sb.setLength(sb.length() - 1); // remove preceding space
                            sb.append(c);
                            i++;
                            continue;
                        }
                    }

                    // Contractions starting with apostrophe
                    if (c == '\'') {
                        // " 's", " 'm", " 'd", " 't" (part of n't)
                        if (c2 == 's' || c2 == 'm' || c2 == 'd') {
                            sb.setLength(sb.length() - 1);
                            sb.append('\'').append(c2);
                            i += 2;
                            continue;
                        }
                        // " 're", " 've", " 'll"
                        if (i + 2 < text.length()) {
                            char c3 = text.charAt(i + 2);
                            if ((c2 == 'r' && c3 == 'e')
                                    || (c2 == 'v' && c3 == 'e')
                                    || (c2 == 'l' && c3 == 'l')
                                    || (c2 == 'n' && c3 == 't')) {
                                sb.setLength(sb.length() - 1);
                                sb.append('\'').append(c2).append(c3);
                                i += 3;
                                continue;
                            }
                        }
                    }
                }
            }

            // Standalone pattern: " ' " → "'"
            if (i > 0
                    && i + 1 < text.length()
                    && text.charAt(i) == '\''
                    && text.charAt(i - 1) == ' '
                    && text.charAt(i + 1) == ' ') {
                sb.setLength(sb.length() - 1); // remove preceding space
                sb.append('\'');
                i += 2; // skip apostrophe and following space
                continue;
            }

            sb.append(text.charAt(i));
            i++;
        }

        return sb.toString();
    }
}
