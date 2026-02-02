package ai.qxotic.jota.panama;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRInput;
import ai.qxotic.jota.ir.lir.ScalarInput;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

final class LIRKernelArgsBuilder {

    KernelArgs build(LIRGraph graph, List<Tensor> inputs, List<MemoryView<?>> outputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        KernelArgs args = new KernelArgs();
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            Tensor tensor = inputs.get(i);
            if (input instanceof ScalarInput scalarInput) {
                long rawBits = readScalarBits(tensor, scalarInput.dataType());
                args.addScalarBits(rawBits, scalarInput.dataType());
                continue;
            }
            MemoryView<?> view = tensor.tryGetMaterialized().orElseGet(tensor::materialize);
            args.addBuffer(view);
        }
        for (MemoryView<?> output : outputs) {
            args.addBuffer(output);
        }
        return args;
    }

    private long readScalarBits(Tensor tensor, DataType type) {
        java.util.OptionalLong constantBits = tensor.scalarConstantBits();
        if (constantBits.isPresent()) {
            return constantBits.getAsLong();
        }

        MemoryView<?> view = tensor.tryGetMaterialized().orElseGet(tensor::materialize);
        if (!(view.memory().base() instanceof MemorySegment segment)) {
            throw new IllegalArgumentException("Scalar input requires MemorySegment backing");
        }
        long offset = view.byteOffset();
        if (type == DataType.FP32) {
            return Float.floatToRawIntBits(segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset));
        }
        if (type == DataType.FP64) {
            return Double.doubleToRawLongBits(
                    segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset));
        }
        if (type == DataType.FP16) {
            short bits = segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
            return Float.floatToFloat16(Float.float16ToFloat(bits));
        }
        if (type == DataType.BF16) {
            short bits = segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
            return BFloat16.fromFloat(BFloat16.toFloat(bits));
        }
        if (type == DataType.I8 || type == DataType.BOOL) {
            return segment.get(ValueLayout.JAVA_BYTE, offset);
        }
        if (type == DataType.I16) {
            return segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        }
        if (type == DataType.I32) {
            return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        }
        if (type == DataType.I64) {
            return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        }
        throw new UnsupportedOperationException("Unsupported scalar dtype: " + type);
    }
}
