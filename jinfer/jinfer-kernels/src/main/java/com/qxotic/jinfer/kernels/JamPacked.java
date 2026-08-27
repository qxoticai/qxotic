package com.qxotic.jinfer.kernels;

import com.qxotic.jota.DataType;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Impl-only weight dtype: the bytes hold jam's packed in-memory layout (jam.h {@code JAM_PACK_ABI})
 * of {@link #base()}, produced by {@link JamPack} at load time. One storage block is one WEIGHT ROW
 * ({@code elementsPerBlock == k}), so all view algebra moves whole rows and can never split the
 * layout; jam additionally requires offsets at 4-row-group boundaries ({@link MatMul#jamApplies}).
 * Never a wire format: {@link GGMLDataTypes} does not know this type, so it cannot be serialized,
 * and only the jam backend can read it.
 */
final class JamPacked implements DataType {

    private final DataType base;
    private final long k;
    private final long rowBytes;

    private JamPacked(DataType base, long k, long rowBytes) {
        this.base = base;
        this.k = k;
        this.rowBytes = rowBytes;
    }

    /** A per-tensor instance: {@code rowBytes = packSize / rows} (uniform by construction). */
    static JamPacked of(DataType base, long k, long rowBytes) {
        return new JamPacked(base, k, rowBytes);
    }

    /** The canonical quantization this packs - the value-preserving identity of the tensor. */
    public DataType base() {
        return base;
    }

    @Override
    public long byteSize() {
        return rowBytes;
    }

    @Override
    public long elementsPerBlock() {
        return k;
    }

    @Override
    public MemoryLayout layout() {
        return MemoryLayout.sequenceLayout(rowBytes, ValueLayout.JAVA_BYTE).withName(name());
    }

    @Override
    public boolean isFloatingPoint() {
        return false;
    }

    @Override
    public boolean isIntegral() {
        return false;
    }

    @Override
    public String name() {
        return base.name() + "+jam";
    }

    @Override
    public List<String> aliases() {
        return List.of();
    }

    @Override
    public String toString() {
        return name();
    }
}
