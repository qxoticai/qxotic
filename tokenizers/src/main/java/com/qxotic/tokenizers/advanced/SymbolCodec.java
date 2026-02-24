package com.qxotic.tokenizers.advanced;

import java.nio.charset.StandardCharsets;

/**
 * Encodes and decodes between UTF-8 bytes and tokenizer symbol strings.
 *
 * <p>This abstraction keeps model-specific byte/symbol logic out of tokenizer algorithms.
 */
public interface SymbolCodec {

    SymbolCodec IDENTITY =
            new SymbolCodec() {
                @Override
                public String encodeBytes(byte[] bytes) {
                    return new String(bytes, StandardCharsets.UTF_8);
                }

                @Override
                public byte[] decodeSymbols(String symbols) {
                    return symbols.getBytes(StandardCharsets.UTF_8);
                }
            };

    SymbolCodec BYTE_LEVEL =
            new SymbolCodec() {
                @Override
                public String encodeBytes(byte[] bytes) {
                    return ByteEncoding.bytesToString(bytes);
                }

                @Override
                public byte[] decodeSymbols(String symbols) {
                    return ByteEncoding.stringToBytes(symbols);
                }
            };

    String encodeBytes(byte[] bytes);

    byte[] decodeSymbols(String symbols);

    default String encodeText(CharSequence text) {
        return encodeBytes(text.toString().getBytes(StandardCharsets.UTF_8));
    }

    default String decodeToText(String symbols) {
        return new String(decodeSymbols(symbols), StandardCharsets.UTF_8);
    }
}
