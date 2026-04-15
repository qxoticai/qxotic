package com.qxotic.toknroll.advanced;

import com.qxotic.toknroll.Vocabulary;

/**
 * Encodes pre-tokenized text chunks into initial BPE token ids and decodes token ids back to raw
 * bytes.
 */
public interface BpeSymbolEncoder {

    int[] encodeChunkToTokenIds(CharSequence chunk, Vocabulary vocabulary);

    default int[] encodeChunkToTokenIds(
            CharSequence source, int startInclusive, int endExclusive, Vocabulary vocabulary) {
        return encodeChunkToTokenIds(source.subSequence(startInclusive, endExclusive), vocabulary);
    }

    default int maxEncodedLength(int charLength) {
        return Math.max(0, charLength) << 2;
    }

    default int encodeChunkToTokenIdsInto(
            CharSequence source,
            int startInclusive,
            int endExclusive,
            Vocabulary vocabulary,
            int[] output) {
        int[] encoded = encodeChunkToTokenIds(source, startInclusive, endExclusive, vocabulary);
        if (output.length < encoded.length) {
            throw new IllegalArgumentException(
                    "Output too small: need " + encoded.length + ", got " + output.length);
        }
        System.arraycopy(encoded, 0, output, 0, encoded.length);
        return encoded.length;
    }

    byte[] decodeTokenBytes(int tokenId, Vocabulary vocabulary);
}
