package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A SentencePiece tokenizer as the text sees it: its vocabulary spells a space U+2581, the text '
 * '.
 *
 * <p>Encoding prepends U+2581 as a Metaspace pre-tokenizer's {@code prepend_scheme} says: before
 * every text ({@code always}), before the start of the text only ({@code first}), or {@code never};
 * and never before text that starts with a space already. Spelling the text's own spaces is the
 * base pipeline's normalizer's work.
 *
 * <p>Decoding writes each token's bytes: a byte token its byte, any other token its spelling with
 * U+2581 a space. With {@code trimLeadingSpace} it drops the space the first token starts with, as
 * a decoder {@code Strip} or a Metaspace decoder drops the prepended one. Text is the UTF-8 of
 * those bytes, so {@link #decode} and {@link #decodeBytes} always agree.
 */
final class MetaspaceTokenizer implements Tokenizer {

    private static final char METASPACE = '\u2581';

    private final Tokenizer base;
    private final boolean prependAlways;
    private final boolean prependFirst;
    private final boolean trimLeadingSpace;
    private final byte[][] tokenBytes;

    MetaspaceTokenizer(Tokenizer base, String prependScheme, boolean trimLeadingSpace) {
        switch (prependScheme) {
            case "always":
            case "first":
            case "never":
                break;
            default:
                throw new IllegalArgumentException("prepend scheme: " + prependScheme);
        }
        this.base = base;
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
        if (prepends(text, start, end)) {
            String prefixed = METASPACE + text.subSequence(start, end).toString();
            base.encodeInto(prefixed, 0, prefixed.length(), out);
        } else {
            base.encodeInto(text, start, end, out);
        }
    }

    @Override
    public int countTokens(CharSequence text, int start, int end) {
        if (prepends(text, start, end)) {
            String prefixed = METASPACE + text.subSequence(start, end).toString();
            return base.countTokens(prefixed, 0, prefixed.length());
        }
        return base.countTokens(text, start, end);
    }

    // ponytail: the space test reads the text before the base's normalizer; a normalizer that
    // turns a first char into a space (NFKC on U+3000) would need it after
    private boolean prepends(CharSequence text, int start, int end) {
        return start < end
                && (prependAlways || prependFirst && start == 0)
                && text.charAt(start) != ' '
                && text.charAt(start) != METASPACE;
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
