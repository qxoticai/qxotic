package com.qxotic.jota.stablehlo.bytecode;

import java.nio.charset.StandardCharsets;

/**
 * Cursor over a slice {@code [start, end)} of a bytecode buffer, decoding the primitives of the
 * MLIR bytecode format ({@code mlir/docs/BytecodeFormat.md}): bytes, prefix varints, zigzag signed
 * varints, flagged varints, null-terminated strings and sections.
 *
 * <p>Alignment is relative to the start of the whole buffer, which MLIR requires to be aligned at
 * least as strictly as every section.
 */
final class EncodingReader {

    /** Padding byte used before aligned sections and blobs. */
    static final int ALIGNMENT_BYTE = 0xCB;

    private final byte[] buf;
    private final int end;
    private int pos;

    EncodingReader(byte[] buf, int start, int end) {
        if (start < 0 || end > buf.length || start > end) {
            throw new BytecodeException("invalid slice [" + start + ", " + end + ")");
        }
        this.buf = buf;
        this.pos = start;
        this.end = end;
    }

    EncodingReader(byte[] buf) {
        this(buf, 0, buf.length);
    }

    /** A reader over {@code [start, end)} of the same buffer. */
    EncodingReader slice(int start, int end) {
        return new EncodingReader(buf, start, end);
    }

    byte[] buffer() {
        return buf;
    }

    int position() {
        return pos;
    }

    int end() {
        return end;
    }

    boolean isEmpty() {
        return pos >= end;
    }

    int remaining() {
        return end - pos;
    }

    BytecodeException error(String message) {
        return new BytecodeException(message + " (at byte offset " + pos + ")");
    }

    int readByte() {
        if (pos >= end) {
            throw error("unexpected end of data");
        }
        return buf[pos++] & 0xFF;
    }

    /** Returns the start offset of {@code length} bytes and skips them. */
    int skipBytes(long length) {
        if (length < 0 || length > remaining()) {
            throw error(
                    "attempting to read " + length + " bytes when only " + remaining() + " remain");
        }
        int start = pos;
        pos += (int) length;
        return start;
    }

    byte[] readBytes(long length) {
        int start = skipBytes(length);
        byte[] out = new byte[(int) length];
        System.arraycopy(buf, start, out, 0, (int) length);
        return out;
    }

    /** PrefixVarInt: trailing zero bits of the first byte count the additional bytes. */
    long readVarInt() {
        int first = readByte();
        if ((first & 1) != 0) {
            return first >>> 1;
        }
        if (first == 0) {
            long v = 0;
            for (int i = 0; i < 8; i++) {
                v |= (long) readByte() << (8 * i);
            }
            return v;
        }
        int extra = Integer.numberOfTrailingZeros(first);
        long v = first;
        for (int i = 1; i <= extra; i++) {
            v |= (long) readByte() << (8 * i);
        }
        return v >>> (extra + 1);
    }

    /** Zigzag-decoded signed varint. */
    long readSignedVarInt() {
        long v = readVarInt();
        return (v >>> 1) ^ -(v & 1);
    }

    int readSize() {
        long v = readVarInt();
        if (v < 0 || v > Integer.MAX_VALUE) {
            throw error("size out of range: " + Long.toUnsignedString(v));
        }
        return (int) v;
    }

    /** {@code (value << 1) | flag}: returns the value, the flag is in {@link #lastFlag()}. */
    long readVarIntWithFlag() {
        long v = readVarInt();
        lastFlag = (v & 1) != 0;
        return v >>> 1;
    }

    private boolean lastFlag;

    boolean lastFlag() {
        return lastFlag;
    }

    String readNullTerminatedString() {
        int start = pos;
        while (pos < end && buf[pos] != 0) {
            pos++;
        }
        if (pos >= end) {
            throw error("malformed null-terminated string");
        }
        String s = new String(buf, start, pos - start, StandardCharsets.ISO_8859_1);
        pos++;
        return s;
    }

    /** Skips 0xCB padding up to an absolute buffer offset multiple of {@code alignment}. */
    void alignTo(long alignment) {
        if (alignment <= 0 || Long.bitCount(alignment) != 1) {
            throw error("alignment must be a power of two, got " + alignment);
        }
        while ((pos & (alignment - 1)) != 0) {
            int padding = readByte();
            if (padding != ALIGNMENT_BYTE) {
                throw error("expected alignment byte 0xCB, got 0x" + Integer.toHexString(padding));
            }
        }
    }

    /** A section: id (low 7 bits), and the reader over its data. */
    record Section(int id, EncodingReader data) {}

    Section readSection() {
        int idAndAlign = readByte();
        long length = readVarInt();
        int id = idAndAlign & 0x7F;
        if ((idAndAlign & 0x80) != 0) {
            alignTo(readVarInt());
        }
        int start = skipBytes(length);
        return new Section(id, slice(start, start + (int) length));
    }

    /** {@code alignment: varint, size: varint, padding, data} (resource blobs). */
    byte[] readBlobAndAlignment() {
        long alignment = readVarInt();
        long size = readVarInt();
        alignTo(alignment);
        return readBytes(size);
    }
}
