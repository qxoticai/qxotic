package com.qxotic.jota.stablehlo.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TIRInterpreter;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Result of a StableHLO → TIR conversion: the graph and, for each graph input (by id), where its
 * data comes from: a function argument, or a non-splat constant hoisted out of the program (TIR has
 * no dense constant node).
 */
public record TirConversion(
        TIRGraph graph, List<InputBinding> inputs, List<TensorType> resultTypes) {

    /** Origin of a TIR graph input. */
    public sealed interface InputBinding {}

    /** Input bound to argument {@code index} of the converted function. */
    public record Argument(int index) implements InputBinding {}

    /** Input bound to a constant of the program. */
    public record Constant(DenseElementsAttr value) implements InputBinding {}

    public TirConversion {
        Objects.requireNonNull(graph, "graph");
        inputs = List.copyOf(inputs);
        resultTypes = List.copyOf(resultTypes);
    }

    /** Graph inputs in id order: arguments as given, constants materialized in {@code domain}. */
    public List<MemoryView<?>> bindInputs(
            List<? extends MemoryView<?>> arguments, MemoryDomain<MemorySegment> domain) {
        List<MemoryView<?>> out = new ArrayList<>(inputs.size());
        for (InputBinding b : inputs) {
            switch (b) {
                case Argument a -> out.add(arguments.get(a.index()));
                case Constant c -> out.add(materialize(c.value(), domain));
            }
        }
        return out;
    }

    /** Runs the graph with {@link TIRInterpreter}. */
    public List<MemoryView<MemorySegment>> interpret(
            List<? extends MemoryView<?>> arguments, MemoryDomain<MemorySegment> domain) {
        return TIRInterpreter.execute(graph, bindInputs(arguments, domain), domain);
    }

    /** A row-major memory view holding {@code value} (splats are expanded). */
    public static MemoryView<MemorySegment> materialize(
            DenseElementsAttr value, MemoryDomain<MemorySegment> domain) {
        TensorType tt = value.type();
        DataType dt = TirTypes.dataType(tt.elementType(), "stablehlo.constant");
        long n = tt.numElements();
        int bytes = (int) dt.byteSize();
        if (bytes != value.elementBytes()) {
            throw new UnsupportedStableHloException(
                    "stablehlo.constant", "element storage mismatch for " + tt.mlir());
        }
        Memory<MemorySegment> memory = domain.memoryAllocator().allocateMemory(dt, n);
        byte[] raw = value.rawData();
        MemorySegment segment = memory.base();
        for (long i = 0; i < n; i++) {
            long from = value.isSplat() ? 0 : i * bytes;
            MemorySegment.copy(raw, (int) from, segment, ValueLayout.JAVA_BYTE, i * bytes, bytes);
        }
        return MemoryView.of(memory, dt, Layout.rowMajor(com.qxotic.jota.Shape.flat(tt.dims())));
    }

    /** Reads a TIR output back as a dense attribute of {@code type}. */
    public static DenseElementsAttr toDense(MemoryView<MemorySegment> view, TensorType type) {
        long n = type.numElements();
        int bytes = (int) view.dataType().byteSize();
        byte[] out = new byte[Math.toIntExact(n * bytes)];
        MemorySegment segment = view.memory().base();
        for (long i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(view, i);
            MemorySegment.copy(
                    segment, ValueLayout.JAVA_BYTE, offset, out, (int) (i * bytes), bytes);
        }
        if (view.dataType() == DataType.BOOL) {
            for (int i = 0; i < out.length; i++) {
                out[i] = (byte) (out[i] != 0 ? 1 : 0);
            }
        }
        return new DenseElementsAttr(type, out, false);
    }
}
