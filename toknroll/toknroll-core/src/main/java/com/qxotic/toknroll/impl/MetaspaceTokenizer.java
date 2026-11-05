package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * A SentencePiece tokenizer as the text sees it: its vocabulary spells a space U+2581, the text '
 * '.
 *
 * <p>Encoding normalizes the text first, then prepends U+2581 as a Metaspace pre-tokenizer's {@code
 * prepend_scheme} says: before every text ({@code always}), before the start of the text only
 * ({@code first}), or {@code never}; and never before text that starts with a space already, one
 * the normalizer made included (NFKC turns U+3000 into one). The base must not normalize again: the
 * loaders give it none and hand the normalizer here.
 *
 * <p>Decoding writes each token's bytes: a byte token its byte, any other token its spelling with
 * U+2581 a space. With {@code trimLeadingSpace} it drops the space the first token starts with, as
 * a decoder {@code Strip} or a Metaspace decoder drops the prepended one. Text is the UTF-8 of
 * those bytes, so {@link #decode} and {@link #decodeBytes} always agree.
 */
final class MetaspaceTokenizer implements Tokenizer {

    private static final char METASPACE = '\u2581';

    private final Tokenizer base;
    private final Normalizer normalizer;
    private final boolean prependAlways;
    private final boolean prependFirst;
    private final boolean trimLeadingSpace;
    private final byte[][] tokenBytes;

    MetaspaceTokenizer(
            Tokenizer base, Normalizer normalizer, String prependScheme, boolean trimLeadingSpace) {
        switch (prependScheme) {
            case "always":
            case "first":
            case "never":
                break;
            default:
                throw new IllegalArgumentException("prepend scheme: " + prependScheme);
        }
        this.base = base;
        this.normalizer = normalizer == null ? Normalizer.identity() : normalizer;
        this.prependAlways = prependScheme.equals("always");
        this.prependFirst = prependScheme.equals("first");
        this.trimLeadingSpace = trimLeadingSpace;
        Vocabulary vocabulary = base.vocabulary();
        this.tokenBytes = new byte[vocabulary.size()][];
        for (int id = 0; id < vocabulary.size(); id++) {
            if (vocabulary.contains(id)) {
                tokenBytes[id] =
                        vocabulary.isTokenOfType(id, StandardTokenType.BYTE)
                                ? base.decodeBytes(IntSequence.of(id)) // its byte, not "<0xC3>"
                                : vocabulary
                                        .token(id)
                                        .replace(METASPACE, ' ')
                                        .getBytes(StandardCharsets.UTF_8);
            }
        }
    }

    @Override
    public Vocabulary vocabulary() {
        return base.vocabulary();
    }

    @Override
    public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
        if (passesThrough(start)) {
            base.encodeInto(text, start, end, out);
        } else {
            CharSequence ready = ready(text.subSequence(start, end), start == 0);
            base.encodeInto(ready, 0, ready.length(), out);
        }
    }

    /** A range of the one text, as {@link #encodeInto}: "first" prepends at its start only. */
    @Override
    public void encodeUtf8Into(byte[] utf8, int start, int end, IntSequence.Builder out) {
        Objects.requireNonNull(utf8, "utf8");
        Objects.checkFromToIndex(start, end, utf8.length);
        if (passesThrough(start)) {
            base.encodeUtf8Into(utf8, start, end, out);
        } else {
            String text = new String(utf8, start, end - start, StandardCharsets.UTF_8);
            CharSequence ready = ready(text, start == 0);
            base.encodeInto(ready, 0, ready.length(), out);
        }
    }

    @Override
    public int countTokens(CharSequence text, int start, int end) {
        if (passesThrough(start)) {
            return base.countTokens(text, start, end);
        }
        CharSequence ready = ready(text.subSequence(start, end), start == 0);
        return base.countTokens(ready, 0, ready.length());
    }

    /** Nothing to normalize, nothing to prepend at this start: the base reads the text as it is. */
    private boolean passesThrough(int start) {
        return normalizer == Normalizer.identity() && !prependsAt(start == 0);
    }

    /** The range normalized, U+2581 before it as the scheme says, unless a space starts it. */
    private CharSequence ready(CharSequence range, boolean textStart) {
        CharSequence normalized = normalizer.apply(range);
        if (normalized.length() == 0 || !prependsAt(textStart)) {
            return normalized;
        }
        char first = normalized.charAt(0);
        return first == ' ' || first == METASPACE ? normalized : METASPACE + normalized.toString();
    }

    private boolean prependsAt(boolean textStart) {
        return prependAlways || prependFirst && textStart;
    }

    @Override
    public String decode(IntSequence tokens) {
        return new String(decodeBytes(tokens), StandardCharsets.UTF_8);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        byte[] out = new byte[countBytes(tokens)];
        decodeBytesInto(tokens, 0, ByteBuffer.wrap(out));
        return out;
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        int length = tokens.length();
        if (tokenStartIndex < 0 || tokenStartIndex > length) {
            throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
        }
        int consumed = 0;
        for (int i = tokenStartIndex; i < length; i++) {
            byte[] bytes = tokenBytes(tokens.intAt(i));
            int skip = skip(i, bytes);
            if (bytes.length - skip > out.remaining()) {
                if (consumed == 0) {
                    throw new IllegalArgumentException("Not enough output space");
                }
                break;
            }
            out.put(bytes, skip, bytes.length - skip);
            consumed++;
        }
        return consumed;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        int total = 0;
        for (int i = 0; i < tokens.length(); i++) {
            byte[] bytes = tokenBytes(tokens.intAt(i));
            total += bytes.length - skip(i, bytes);
        }
        return total;
    }

    @Override
    public float expectedTokensPerChar() {
        return base.expectedTokensPerChar();
    }

    /** The bytes to drop from token {@code index}: the first token's leading space, if trimmed. */
    private int skip(int index, byte[] bytes) {
        return index == 0 && trimLeadingSpace && bytes.length > 0 && bytes[0] == ' ' ? 1 : 0;
    }

    private byte[] tokenBytes(int id) {
        if (id < 0 || id >= tokenBytes.length || tokenBytes[id] == null) {
            throw new IllegalArgumentException("Unknown token id: " + id);
        }
        return tokenBytes[id];
    }
}
