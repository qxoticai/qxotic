package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Vocabulary;
import java.nio.charset.StandardCharsets;

/**
 * Symbol encoder that treats each Unicode code point as a base symbol.
 *
 * <p>Useful for non-byte-level BPE vocabularies where base symbols are code points.
 */
public final class CodePointSymbolEncoder implements BpeSymbolEncoder {

    @Override
    public int[] encodeChunkToTokenIds(CharSequence chunk, Vocabulary vocabulary) {
        if (chunk.length() == 0) {
            return new int[0];
        }

        int cpCount = Character.codePointCount(chunk, 0, chunk.length());
        int[] out = new int[cpCount];
        int outIdx = 0;
        for (int i = 0; i < chunk.length(); ) {
            int cp = Character.codePointAt(chunk, i);
            String symbol = new String(Character.toChars(cp));
            out[outIdx++] = vocabulary.id(symbol);
            i += Character.charCount(cp);
        }
        return out;
    }

    @Override
    public byte[] decodeTokenBytes(int tokenId, Vocabulary vocabulary) {
        return vocabulary.token(tokenId).getBytes(StandardCharsets.UTF_8);
    }
}
