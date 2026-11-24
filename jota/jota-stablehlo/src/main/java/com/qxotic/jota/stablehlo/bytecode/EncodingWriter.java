package com.qxotic.jota.stablehlo.bytecode;

import java.io.ByteArrayOutputStream;

/** Growable byte buffer with the primitive encodings of the MLIR bytecode format. */
final class EncodingWriter {

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    int size() {
        return out.size();
    }

    byte[] toByteArray() {
        return out.toByteArray();
    }

    void writeByte(int b) {
        out.write(b & 0xFF);
    }

    void writeBytes(byte[] bytes) {
        out.write(bytes, 0, bytes.length);
    }

    /** PrefixVarInt: the trailing zero bits of the first byte count the additional bytes. */
    void writeVarInt(long value) {
        for (int bytes = 1; bytes <= 8; bytes++) {
            if (Long.compareUnsigned(value, 1L << (7 * bytes)) < 0
                    || (bytes == 8 && value >>> 56 == 0)) {
                long encoded = (value << bytes) | (1L << (bytes - 1));
                for (int i = 0; i < bytes; i++) {
                    writeByte((int) (encoded >>> (8 * i)));
                }
                return;
            }
        }
        writeByte(0); // 9 bytes: a zero prefix byte then the full little-endian value
        for (int i = 0; i < 8; i++) {
            writeByte((int) (value >>> (8 * i)));
        }
    }

    /** Zigzag encoding: {@code (value << 1) ^ (value >> 63)}. */
    void writeSignedVarInt(long value) {
        writeVarInt((value << 1) ^ (value >> 63));
    }

    void writeVarIntWithFlag(long value, boolean flag) {
        writeVarInt((value << 1) | (flag ? 1 : 0));
    }

    /** {@code size: varint, bytes}. */
    void writeBlob(byte[] bytes) {
        writeVarInt(bytes.length);
        writeBytes(bytes);
    }

    /** A section: id byte (no alignment), length, data. */
    void writeSection(int id, byte[] data) {
        writeByte(id);
        writeVarInt(data.length);
        writeBytes(data);
    }

    void writeNullTerminatedString(String s) {
        for (int i = 0; i < s.length(); i++) {
            writeByte(s.charAt(i));
        }
        writeByte(0);
    }
}
