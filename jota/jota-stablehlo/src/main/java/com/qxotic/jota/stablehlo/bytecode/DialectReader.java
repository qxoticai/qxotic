package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * What a dialect decoder sees (MLIR's {@code DialectBytecodeReader}): primitives plus references
 * into the string, attribute and type tables.
 */
final class DialectReader {

    private final MlirBytecodeReader owner;
    private final EncodingReader r;

    DialectReader(MlirBytecodeReader owner, EncodingReader r) {
        this.owner = owner;
        this.r = r;
    }

    BytecodeException error(String message) {
        return r.error(message);
    }

    long bytecodeVersion() {
        return owner.version();
    }

    long varInt() {
        return r.readVarInt();
    }

    int size() {
        return r.readSize();
    }

    long signedVarInt() {
        return r.readSignedVarInt();
    }

    int readByte() {
        return r.readByte();
    }

    boolean bool() {
        return r.readByte() != 0;
    }

    Attribute attribute() {
        return owner.attribute(r.readVarInt());
    }

    /**
     * {@code (index << 1) | present}; null when absent. MLIR writes an absent attribute as 0, so a
     * clear flag with an index is rejected (it means the data is read with the wrong layout).
     */
    Attribute optionalAttribute() {
        long index = r.readVarIntWithFlag();
        if (!r.lastFlag()) {
            if (index != 0) {
                throw error("absent optional attribute with index " + index);
            }
            return null;
        }
        return owner.attribute(index);
    }

    Type type() {
        return owner.type(r.readVarInt());
    }

    String string() {
        return owner.string(r.readVarInt());
    }

    /** {@code size: varint, bytes}. */
    byte[] blob() {
        return r.readBytes(r.readVarInt());
    }

    /**
     * APInt of known bit width: one byte up to 8 bits, a zigzag varint up to 64 bits (raw bits,
     * truncated to the width).
     */
    long apInt(int bitWidth) {
        long v;
        if (bitWidth <= 8) {
            v = r.readByte();
        } else if (bitWidth <= 64) {
            v = r.readSignedVarInt();
        } else {
            throw r.error("integers wider than 64 bits are not supported (" + bitWidth + ")");
        }
        return bitWidth >= 64 ? v : v & ((1L << bitWidth) - 1);
    }

    <T> List<T> list(Supplier<T> element) {
        int n = r.readSize();
        List<T> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(element.get());
        }
        return out;
    }

    long[] signedVarInts() {
        int n = r.readSize();
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = r.readSignedVarInt();
        }
        return out;
    }
}
