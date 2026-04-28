package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * Reusable decode wrapper that replaces a metaspace marker with space and optionally trims a
 * leading space from the decoded output.
 *
 * <p>Subclasses override {@link #transformDecoded(String, boolean)} to apply the metaspace-to-space
 * conversion and {@link #trimLeadingSpaceAtStart()} to indicate whether a leading space artifact
 * should be stripped from byte-level decode.
 */
public abstract class TransformedTokenizer implements Tokenizer {

    private static final char METASPACE = '\u2581';
    private static final byte SPACE_BYTE = (byte) ' ';

    protected final Tokenizer base;
    private final byte[][] transformedTokenBytes;
    private final boolean[] byteFallbackToken;

    protected TransformedTokenizer(Tokenizer base) {
        this.base = base;
        Vocabulary vocabulary = base.vocabulary();
        this.transformedTokenBytes = new byte[vocabulary.size()][];
        this.byteFallbackToken = new boolean[vocabulary.size()];
        for (int id = 0; id < vocabulary.size(); id++) {
            if (!vocabulary.contains(id)) {
                continue;
            }
            transformedTokenBytes[id] =
                    vocabulary.token(id).replace(METASPACE, ' ').getBytes(StandardCharsets.UTF_8);
            byteFallbackToken[id] = vocabulary.isTokenOfType(id, StandardTokenType.BYTE);
        }
    }

    protected abstract String transformDecoded(String decoded, boolean atStartOfText);

    protected boolean trimLeadingSpaceAtStart() {
        return false;
    }

    @Override
    public Vocabulary vocabulary() {
        return base.vocabulary();
    }

    @Override
    public String decode(IntSequence tokens) {
        return transformDecoded(base.decode(tokens), true);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        if (hasByteFallback(tokens, 0)) {
            return decode(tokens).getBytes(StandardCharsets.UTF_8);
        }
        int total = countBytes(tokens);
        byte[] out = new byte[total];
        int pos = 0;
        for (int i = 0; i < tokens.length(); i++) {
            byte[] bytes = tokenBytes(tokens.intAt(i));
            int start = (i == 0 && trimLeadingSpaceAtStart() && startsWithSpace(bytes)) ? 1 : 0;
            int len = bytes.length - start;
            if (len > 0) {
                System.arraycopy(bytes, start, out, pos, len);
                pos += len;
            }
        }
        return out;
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        int length = tokens.length();
        if (tokenStartIndex < 0 || tokenStartIndex > length) {
            throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
        }
        if (tokenStartIndex == length) {
            return 0;
        }
        if (hasByteFallback(tokens, tokenStartIndex)) {
            return decodeBytesIntoFallback(tokens, tokenStartIndex, out);
        }

        int consumed = 0;
        for (int i = tokenStartIndex; i < length; i++) {
            byte[] bytes = tokenBytes(tokens.intAt(i));
            int start =
                    (i == tokenStartIndex
                                    && tokenStartIndex == 0
                                    && trimLeadingSpaceAtStart()
                                    && startsWithSpace(bytes))
                            ? 1
                            : 0;
            int len = bytes.length - start;
            if (consumed == 0 && len > out.remaining()) {
                throw new IllegalArgumentException("Not enough output space");
            }
            if (len > out.remaining()) {
                break;
            }
            if (len > 0) {
                out.put(bytes, start, len);
            }
            consumed++;
        }
        return consumed;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        if (hasByteFallback(tokens, 0)) {
            return decode(tokens).getBytes(StandardCharsets.UTF_8).length;
        }
        int total = 0;
        for (int i = 0; i < tokens.length(); i++) {
            byte[] bytes = tokenBytes(tokens.intAt(i));
            total += bytes.length;
            if (i == 0 && trimLeadingSpaceAtStart() && startsWithSpace(bytes)) {
                total--;
            }
        }
        return total;
    }

    @Override
    public float expectedTokensPerChar() {
        return base.expectedTokensPerChar();
    }

    public static String normalizeMetaspaceDecoded(String decoded, boolean trimLeadingSpace) {
        String normalized = decoded.replace(METASPACE, ' ');
        if (trimLeadingSpace && normalized.length() > 0 && normalized.charAt(0) == ' ') {
            return normalized.substring(1);
        }
        return normalized;
    }

    private byte[] tokenBytes(int tokenId) {
        if (tokenId < 0
                || tokenId >= transformedTokenBytes.length
                || transformedTokenBytes[tokenId] == null) {
            throw new IllegalArgumentException("Unknown token id: " + tokenId);
        }
        return transformedTokenBytes[tokenId];
    }

    private boolean hasByteFallback(IntSequence tokens, int startIndex) {
        for (int i = startIndex; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            if (tokenId >= 0 && tokenId < byteFallbackToken.length && byteFallbackToken[tokenId]) {
                return true;
            }
        }
        return false;
    }

    private int decodeBytesIntoFallback(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        int length = tokens.length();
        boolean atStartOfText = tokenStartIndex == 0;
        int remaining = out.remaining();
        int bestEndExclusive = -1;
        byte[] bestFit = null;

        for (int endExclusive = tokenStartIndex + 1; endExclusive <= length; endExclusive++) {
            byte[] bytes = transformedBytes(tokens, tokenStartIndex, endExclusive, atStartOfText);
            if (bytes.length > remaining) {
                break;
            }
            bestFit = bytes;
            bestEndExclusive = endExclusive;
        }

        if (bestEndExclusive < 0) {
            throw new IllegalArgumentException("Not enough output space");
        }
        out.put(bestFit);
        return bestEndExclusive - tokenStartIndex;
    }

    private byte[] transformedBytes(
            IntSequence tokens, int startInclusive, int endExclusive, boolean atStartOfText) {
        String decoded = base.decode(tokens.subSequence(startInclusive, endExclusive));
        return transformDecoded(decoded, atStartOfText).getBytes(StandardCharsets.UTF_8);
    }

    private static boolean startsWithSpace(byte[] bytes) {
        return bytes.length > 0 && bytes[0] == SPACE_BYTE;
    }
}
