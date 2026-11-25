package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.List;
import java.util.function.Consumer;

/**
 * What a dialect encoder writes into (MLIR's {@code DialectBytecodeWriter}): primitives plus
 * references to strings, attributes and types of the enclosing {@link MlirBytecodeWriter}.
 */
final class AttrTypeWriter {

    private final MlirBytecodeWriter owner;
    private final EncodingWriter out = new EncodingWriter();
    private boolean custom = true;

    AttrTypeWriter(MlirBytecodeWriter owner) {
        this.owner = owner;
    }

    byte[] bytes() {
        return out.toByteArray();
    }

    boolean custom() {
        return custom;
    }

    /** Assembly-format fallback: the entry is the null-terminated textual form. */
    void assembly(String text) {
        custom = false;
        out.writeNullTerminatedString(text);
    }

    void varInt(long v) {
        out.writeVarInt(v);
    }

    void signedVarInt(long v) {
        out.writeSignedVarInt(v);
    }

    void varIntWithFlag(long v, boolean flag) {
        out.writeVarIntWithFlag(v, flag);
    }

    void writeByte(int b) {
        out.writeByte(b);
    }

    void attribute(Attribute a, String codec) {
        out.writeVarInt(owner.attribute(a, codec));
    }

    /** {@code (index << 1) | present}. */
    void optionalAttribute(Attribute a, String codec) {
        if (a == null) {
            out.writeVarIntWithFlag(0, false);
        } else {
            out.writeVarIntWithFlag(owner.attribute(a, codec), true);
        }
    }

    void type(Type t, String codec) {
        out.writeVarInt(owner.type(t, codec));
    }

    void string(String s) {
        out.writeVarInt(owner.string(s));
    }

    void blob(byte[] bytes) {
        out.writeBlob(bytes);
    }

    /** APInt of known width: one byte up to 8 bits, a zigzag varint of the raw bits up to 64. */
    void apInt(long bits, int bitWidth) {
        if (bitWidth <= 8) {
            out.writeByte((int) bits);
        } else if (bitWidth <= 64) {
            long masked = bitWidth == 64 ? bits : bits & ((1L << bitWidth) - 1);
            out.writeSignedVarInt(masked);
        } else {
            throw new BytecodeException("integers wider than 64 bits are not supported");
        }
    }

    <T> void list(List<T> items, Consumer<T> element) {
        out.writeVarInt(items.size());
        for (T item : items) {
            element.accept(item);
        }
    }

    void signedVarInts(long[] values) {
        out.writeVarInt(values.length);
        for (long v : values) {
            out.writeSignedVarInt(v);
        }
    }
}
