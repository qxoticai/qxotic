package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.clike.CLikeDataTypeModel;
import com.qxotic.jota.runtime.clike.CLikeDialect;
import com.qxotic.jota.runtime.clike.CLikeKernelSignatureSupport;
import com.qxotic.jota.runtime.clike.CLikeLirSourceGenerator;
import com.qxotic.jota.runtime.clike.CLikeParallelLoopSupport.LinearIdSpec;
import java.util.Map;

final class HipDialect implements CLikeDialect {

    private static final CLikeDataTypeModel TYPE_MODEL =
            new CLikeDataTypeModel(
                    Map.of(
                            DataType.BOOL, "uint8_t",
                            DataType.I8, "int8_t",
                            DataType.I16, "int16_t",
                            DataType.I32, "int32_t",
                            DataType.I64, "int64_t",
                            DataType.FP16, "__half",
                            DataType.BF16, "hip_bfloat16",
                            DataType.FP32, "float",
                            DataType.FP64, "double"),
                    Map.of(
                            DataType.I8, "uint8_t",
                            DataType.I16, "uint16_t",
                            DataType.I32, "uint32_t",
                            DataType.I64, "uint64_t"));

    @Override
    public String language() {
        return "hip";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return new HipEmitter(graph, scratchLayout, kernelName).generate();
    }

    private static final class HipEmitter extends CLikeLirSourceGenerator {
        private HipEmitter(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            super(graph, scratchLayout, kernelName);
        }

        @Override
        protected void appendPreamble(StringBuilder source) {
            source.append("#include <hip/hip_runtime.h>\n");
            source.append("#include <hip/hip_fp16.h>\n");
            source.append("#include <hip/hip_bfloat16.h>\n");
            source.append("#include <stdint.h>\n");
            source.append("#include <math.h>\n\n");
        }

        @Override
        protected String kernelDeclaration() {
            return "extern \"C\" __global__ void "
                    + kernelName()
                    + "("
                    + renderKernelSignature()
                    + ")";
        }

        @Override
        protected void emitProlog() {
            emitInputs(this::bindInput);
            emitOutputs(
                    (output, outputIndex) -> registerBuffer(output, outputBufferName(outputIndex)));
            addLine("uint8_t *scratch = (uint8_t *)(uintptr_t)scratch_ptr;");
            emitScratchBindings(this::bindScratch);
        }

        private void bindInput(LIRInput input, int inputIndex) {
            if (input instanceof BufferRef buffer) {
                bindReadOnlyInputBuffer(buffer, inputIndex);
                return;
            }
            ScalarInput scalar = requireScalarInput(input);
            bindScalarInput(scalar, inputIndex, "scalar" + inputIndex);
        }

        private void bindScratch(BufferRef scratchBuffer, long offset, int scratchSlot) {
            String name = scratchBufferName(scratchSlot);
            String type = renderDataTypeToken(scratchBuffer.dataType());
            registerBuffer(scratchBuffer, name);
            addLine(type + " *" + name + " = (" + type + " *)(scratch + " + offset + "LL);");
        }

        @Override
        protected boolean tryEmitParallelTopLevel(LIRExprNode body) {
            return emitParallelTopLevel(body);
        }

        @Override
        protected String serialFallbackGuardCondition() {
            return "blockIdx.x == 0 && threadIdx.x == 0";
        }

        @Override
        protected LinearIdSpec linearIdSpec() {
            return new LinearIdSpec(
                    "long long",
                    "(long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x",
                    "1LL");
        }

        @Override
        protected String indexTypeName() {
            return "long long";
        }

        @Override
        protected String indexLiteral(long value) {
            return value + "LL";
        }

        @Override
        protected String renderDataTypeToken(DataType dataType) {
            return TYPE_MODEL.renderType(dataType);
        }

        @Override
        protected String renderScalarLiteralFromBits(long bits, DataType type) {
            return renderLiteralFromTypeModel(TYPE_MODEL, bits, type);
        }

        @Override
        protected String renderCastExpression(DataType source, DataType target, String expr) {
            return renderStandardCastExpression(source, target, expr);
        }

        @Override
        protected String renderFloat32ConversionExpression(DataType source, String expr) {
            return TYPE_MODEL.renderToFloat32(source, expr);
        }

        @Override
        protected String renderFloat64ConversionExpression(DataType source, String expr) {
            return TYPE_MODEL.renderToFloat64(source, expr);
        }

        @Override
        protected String renderBufferReadExpression(
                BufferBinding buffer, DataType type, String offset) {
            String ptr = byteOffsetPointer("uint8_t*", buffer.name, offset);
            return derefRead(renderDataTypeToken(type) + "*", ptr);
        }

        @Override
        protected String renderBufferWriteStatement(
                BufferBinding buffer,
                DataType type,
                String offset,
                String value,
                DataType valueType) {
            String ptr = byteOffsetPointer("uint8_t*", buffer.name, offset);
            String storeValue = storeValueExpr(type, value, valueType);
            return derefWrite(renderDataTypeToken(type) + "*", ptr, storeValue);
        }

        @Override
        protected String unsignedShiftCarrierTypeName(DataType type) {
            return TYPE_MODEL.renderUnsignedShiftCarrierType(type);
        }

        @Override
        protected String float32MathFunctionName(String base) {
            return base + "f";
        }

        @Override
        protected String float64MathFunctionName(String base) {
            return base;
        }

        private String renderKernelSignature() {
            return CLikeKernelSignatureSupport.renderKernelArgumentListGrouped(
                    graph(),
                    this::renderKernelInputArgument,
                    this::renderKernelOutputArgument,
                    new String[] {"uint64_t scratch_ptr"});
        }

        private String renderKernelInputArgument(LIRInput input, int argIndex) {
            if (input instanceof BufferRef buffer) {
                return "const " + renderDataTypeToken(buffer.dataType()) + " *input" + argIndex;
            }
            ScalarInput scalar = requireScalarInput(input);
            return renderDataTypeToken(scalar.dataType()) + " scalar" + argIndex;
        }

        private String renderKernelOutputArgument(
                BufferRef output, int outputIndex, int ignoredArgumentSlot) {
            return renderDataTypeToken(output.dataType()) + " *output" + outputIndex;
        }
    }
}
