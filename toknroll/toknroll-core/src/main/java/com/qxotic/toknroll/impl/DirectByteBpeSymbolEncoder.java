package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.SymbolCodec;
import java.util.Arrays;

/**
 * Low-allocation byte-level symbol encoder for TikToken-style BPE vocabularies.
 *
 * <p>Encodes text ranges directly to UTF-8 bytes and maps each byte to a singleton token id.
 */
public final class DirectByteBpeSymbolEncoder implements BpeSymbolEncoder {

    private static final byte REPLACEMENT_B0 = (byte) 0xEF;
    private static final byte REPLACEMENT_B1 = (byte) 0xBF;
    private static final byte REPLACEMENT_B2 = (byte) 0xBD;

    private final int[] byteTokenId;

    public DirectByteBpeSymbolEncoder(Vocabulary vocabulary) {
        this.byteTokenId = buildByteTokenMap(vocabulary);
    }

    @Override
    public int[] encodeChunkToTokenIds(CharSequence chunk, Vocabulary vocabulary) {
        return encodeChunkToTokenIds(chunk, 0, chunk.length(), vocabulary);
    }

    @Override
    public int[] encodeChunkToTokenIds(
            CharSequence source, int startInclusive, int endExclusive, Vocabulary vocabulary) {
        if (startInclusive >= endExclusive) {
            return new int[0];
        }

        int maxLen = (endExclusive - startInclusive) << 2;
        int[] out = new int[maxLen];
        int tokenLen = utf8EncodeTokenIdsRange(source, startInclusive, endExclusive, out);
        return tokenLen == out.length ? out : Arrays.copyOf(out, tokenLen);
    }

    @Override
    public int maxEncodedLength(int charLength) {
        return Math.max(0, charLength) << 2;
    }

    @Override
    public int encodeChunkToTokenIdsInto(
            CharSequence source,
            int startInclusive,
            int endExclusive,
            Vocabulary vocabulary,
            int[] output) {
        if (startInclusive >= endExclusive) {
            return 0;
        }

        int needed = maxEncodedLength(endExclusive - startInclusive);
        if (output.length < needed) {
            throw new IllegalArgumentException(
                    "Output too small: need " + needed + ", got " + output.length);
        }

        return utf8EncodeTokenIdsRange(source, startInclusive, endExclusive, output);
    }

    @Override
    public byte[] decodeTokenBytes(int tokenId, Vocabulary vocabulary) {
        return SymbolCodec.BYTE_LEVEL.decodeSymbols(vocabulary.token(tokenId));
    }

    private int utf8EncodeTokenIdsRange(
            CharSequence source, int startInclusive, int endExclusive, int[] dst) {
        int dp = 0;
        for (int i = startInclusive; i < endExclusive; i++) {
            char c = source.charAt(i);
            if (c < 0x80) {
                dst[dp++] = byteTokenId[c];
                continue;
            }
            if (c < 0x800) {
                dst[dp++] = byteTokenId[0xC0 | (c >>> 6)];
                dst[dp++] = byteTokenId[0x80 | (c & 0x3F)];
                continue;
            }
            if (Character.isHighSurrogate(c)) {
                if (i + 1 < endExclusive) {
                    char low = source.charAt(i + 1);
                    if (Character.isLowSurrogate(low)) {
                        int cp = Character.toCodePoint(c, low);
                        dst[dp++] = byteTokenId[0xF0 | (cp >>> 18)];
                        dst[dp++] = byteTokenId[0x80 | ((cp >>> 12) & 0x3F)];
                        dst[dp++] = byteTokenId[0x80 | ((cp >>> 6) & 0x3F)];
                        dst[dp++] = byteTokenId[0x80 | (cp & 0x3F)];
                        i++;
                        continue;
                    }
                }
                dst[dp++] = byteTokenId[REPLACEMENT_B0 & 0xFF];
                dst[dp++] = byteTokenId[REPLACEMENT_B1 & 0xFF];
                dst[dp++] = byteTokenId[REPLACEMENT_B2 & 0xFF];
                continue;
            }
            if (Character.isLowSurrogate(c)) {
                dst[dp++] = byteTokenId[REPLACEMENT_B0 & 0xFF];
                dst[dp++] = byteTokenId[REPLACEMENT_B1 & 0xFF];
                dst[dp++] = byteTokenId[REPLACEMENT_B2 & 0xFF];
                continue;
            }
            dst[dp++] = byteTokenId[0xE0 | (c >>> 12)];
            dst[dp++] = byteTokenId[0x80 | ((c >>> 6) & 0x3F)];
            dst[dp++] = byteTokenId[0x80 | (c & 0x3F)];
        }
        return dp;
    }

    private static int[] buildByteTokenMap(Vocabulary vocabulary) {
        int[] map = new int[256];
        Arrays.fill(map, -1);
        for (int i = 0; i < 256; i++) {
            String symbol = SymbolCodec.BYTE_LEVEL.encodeBytes(new byte[] {(byte) i});
            map[i] = vocabulary.id(symbol);
        }
        return map;
    }
}
