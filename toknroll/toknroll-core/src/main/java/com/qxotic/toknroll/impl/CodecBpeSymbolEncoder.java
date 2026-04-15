package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/** Symbol encoder based on a {@link SymbolCodec}. */
public final class CodecBpeSymbolEncoder implements BpeSymbolEncoder {

    private static final byte[] REPLACEMENT_BYTES = {(byte) 0xEF, (byte) 0xBF, (byte) 0xBD};

    private final SymbolCodec symbolCodec;

    public CodecBpeSymbolEncoder(SymbolCodec symbolCodec) {
        this.symbolCodec = Objects.requireNonNull(symbolCodec, "symbolCodec");
    }

    @Override
    public int[] encodeChunkToTokenIds(CharSequence chunk, Vocabulary vocabulary) {
        if (chunk.length() == 0) {
            return new int[0];
        }

        byte[] rawBytes = getBytesWithReplacement(chunk);
        String encodedSymbols = symbolCodec.encodeBytes(rawBytes);
        int[] tokens = new int[encodedSymbols.length()];
        for (int i = 0; i < encodedSymbols.length(); i++) {
            tokens[i] = vocabulary.id(String.valueOf(encodedSymbols.charAt(i)));
        }
        return tokens;
    }

    @Override
    public byte[] decodeTokenBytes(int tokenId, Vocabulary vocabulary) {
        return symbolCodec.decodeSymbols(vocabulary.token(tokenId));
    }

    private static byte[] getBytesWithReplacement(CharSequence text) {
        CharsetEncoder encoder =
                StandardCharsets.UTF_8
                        .newEncoder()
                        .onMalformedInput(CodingErrorAction.REPLACE)
                        .onUnmappableCharacter(CodingErrorAction.REPLACE)
                        .replaceWith(REPLACEMENT_BYTES);
        try {
            ByteBuffer byteBuffer = encoder.encode(CharBuffer.wrap(text));
            byte[] bytes = new byte[byteBuffer.limit()];
            byteBuffer.get(bytes);
            return bytes;
        } catch (CharacterCodingException e) {
            throw new IllegalStateException("UTF-8 encoding failed", e);
        }
    }
}
