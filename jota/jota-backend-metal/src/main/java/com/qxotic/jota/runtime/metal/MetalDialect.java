package com.qxotic.jota.runtime.metal;

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

final class MetalDialect implements CLikeDialect {

    private static final CLikeDataTypeModel TYPE_MODEL =
            new CLikeDataTypeModel(
                    Map.of(
                            DataType.BOOL, "uchar",
                            DataType.I8, "char",
                            DataType.I16, "short",
                            DataType.I32, "int",
                            DataType.I64, "long",
                            DataType.FP16, "half",
                            DataType.BF16, "ushort",
                            DataType.FP32, "float",
                            DataType.FP64, "double"),
                    Map.of(
                            DataType.I8, "uchar",
                            DataType.I16, "ushort",
                            DataType.I32, "uint",
                            DataType.I64, "ulong"));

    @Override
    public String language() {
        return "metal";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return new MetalEmitter(graph, scratchLayout, kernelName).generate();
    }

    private static final class MetalEmitter extends CLikeLirSourceGenerator {
        private MetalEmitter(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            super(graph, scratchLayout, kernelName);
        }

        @Override
        protected void appendPreamble(StringBuilder source) {
            source.append("#include <metal_stdlib>\n");
            source.append("using namespace metal;\n\n");
            source.append("inline float jota_bf16_to_float(ushort bits) {\n");
            source.append("  return as_type<float>(((uint)bits) << 16);\n");
            source.append("}\n");
            source.append("inline ushort jota_float_to_bf16(float v) {\n");
            source.append("  return (ushort)(as_type<uint>(v) >> 16);\n");
            source.append("}\n\n");
        }

        @Override
        protected String kernelDeclaration() {
            return "kernel void " + kernelName() + "(" + renderKernelSignature() + ")";
        }

        @Override
        protected void emitProlog() {
            emitInputs(this::bindInput);
            emitOutputs(
                    (output, outputIndex) -> registerBuffer(output, outputBufferName(outputIndex)));
            emitScratchBindings(this::bindScratch);
        }

        private void bindInput(LIRInput input, int inputIndex) {
            if (input instanceof BufferRef buffer) {
                bindReadOnlyInputBuffer(buffer, inputIndex);
                return;
            }
            ScalarInput scalar = requireScalarInput(input);
            bindScalarInput(scalar, inputIndex, "scalarPtr" + inputIndex + "[0]");
        }

        private void bindScratch(BufferRef scratchBuffer, long offset, int scratchSlot) {
            String name = scratchBufferName(scratchSlot);
            String type = renderDataTypeToken(scratchBuffer.dataType());
            registerBuffer(scratchBuffer, name);
            addLine(
                    "device "
                            + type
                            + " *"
                            + name
                            + " = (device "
                            + type
                            + " *)(scratch + "
                            + offset
                            + "L);");
        }

        @Override
        protected boolean tryEmitParallelTopLevel(LIRExprNode body) {
            return emitParallelTopLevel(body);
        }

        @Override
        protected String serialFallbackGuardCondition() {
            return "gid.x == 0u && gid.y == 0u && gid.z == 0u";
        }

        @Override
        protected LinearIdSpec linearIdSpec() {
            return new LinearIdSpec("long", "(long)gid.x", "1L");
        }

        @Override
        protected String indexTypeName() {
            return "long";
        }

        @Override
        protected String indexLiteral(long value) {
            return value + "L";
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
            if (source == DataType.FP32 && target == DataType.BF16) {
                return "jota_float_to_bf16(" + expr + ")";
            }
            if (source == DataType.BF16 && target == DataType.FP32) {
                return "jota_bf16_to_float(" + expr + ")";
            }
            if (target == DataType.BF16) {
                return "jota_float_to_bf16("
                        + renderFloat32ConversionExpression(source, expr)
                        + ")";
            }
            return renderStandardCastExpression(source, target, expr);
        }

        @Override
        protected String renderFloat32ConversionExpression(DataType source, String expr) {
            if (source == DataType.BF16) {
                return "jota_bf16_to_float(" + expr + ")";
            }
            return TYPE_MODEL.renderToFloat32(source, expr);
        }

        @Override
        protected String renderFloat64ConversionExpression(DataType source, String expr) {
            if (source == DataType.BF16) {
                return "(double)jota_bf16_to_float(" + expr + ")";
            }
            return TYPE_MODEL.renderToFloat64(source, expr);
        }

        @Override
        protected String renderBufferReadExpression(
                BufferBinding buffer, DataType type, String offset) {
            String space = buffer.readOnly ? "const device" : "device";
            String ptr = byteOffsetPointer(space + " uchar*", buffer.name, offset);
            return derefRead(space + " " + renderDataTypeToken(type) + "*", ptr);
        }

        @Override
        protected String renderBufferWriteStatement(
                BufferBinding buffer,
                DataType type,
                String offset,
                String value,
                DataType valueType) {
            if (buffer.readOnly) {
                throw new IllegalStateException("Attempting to write to read-only buffer");
            }
            String ptr = byteOffsetPointer("device uchar*", buffer.name, offset);
            String storeValue = storeValueExpr(type, value, valueType);
            return derefWrite("device " + renderDataTypeToken(type) + "*", ptr, storeValue);
        }

        @Override
        protected String unsignedShiftCarrierTypeName(DataType type) {
            return TYPE_MODEL.renderUnsignedShiftCarrierType(type);
        }

        @Override
        protected String float32MathFunctionName(String base) {
            return base;
        }

        @Override
        protected String float64MathFunctionName(String base) {
            return base;
        }

        private String renderKernelSignature() {
            int scratchSlot = graph().inputs().size() + graph().outputs().size();
            String scratchArg = "device uchar *scratch [[buffer(" + scratchSlot + ")]]";
            return CLikeKernelSignatureSupport.renderKernelArgumentList(
                    graph(),
                    this::renderKernelInputArgument,
                    this::renderKernelOutputArgument,
                    scratchArg,
                    "uint3 gid [[thread_position_in_grid]]");
        }

        private String renderKernelInputArgument(LIRInput input, int argIndex) {
            if (input instanceof BufferRef buffer) {
                return "const device "
                        + renderDataTypeToken(buffer.dataType())
                        + " *input"
                        + argIndex
                        + " [[buffer("
                        + argIndex
                        + ")]]";
            }
            ScalarInput scalar = requireScalarInput(input);
            return "constant "
                    + renderDataTypeToken(scalar.dataType())
                    + " *scalarPtr"
                    + argIndex
                    + " [[buffer("
                    + argIndex
                    + ")]]";
        }

        private String renderKernelOutputArgument(
                BufferRef output, int outputIndex, int argumentSlot) {
            return "device "
                    + renderDataTypeToken(output.dataType())
                    + " *output"
                    + outputIndex
                    + " [[buffer("
                    + argumentSlot
                    + ")]]";
        }
    }
}
