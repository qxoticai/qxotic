package com.qxotic.jota.runtime.opencl;

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

final class OpenClDialect implements CLikeDialect {

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
        return "opencl";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return new OpenClEmitter(graph, scratchLayout, kernelName).generate();
    }

    private static final class OpenClEmitter extends CLikeLirSourceGenerator {
        private OpenClEmitter(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            super(graph, scratchLayout, kernelName);
        }

        @Override
        protected void appendPreamble(StringBuilder source) {
            source.append("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
            source.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n");
            source.append("inline float jota_bf16_to_float(ushort bits) {\n");
            source.append("  return as_float(((uint)bits) << 16);\n");
            source.append("}\n");
            source.append("inline ushort jota_float_to_bf16(float v) {\n");
            source.append("  return (ushort)(as_uint(v) >> 16);\n");
            source.append("}\n\n");
        }

        @Override
        protected String kernelDeclaration() {
            return "__kernel void " + kernelName() + "(" + renderKernelSignature() + ")";
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
            bindScalarInput(scalar, inputIndex, "scalar" + inputIndex);
        }

        private void bindScratch(BufferRef scratchBuffer, long offset, int scratchSlot) {
            String name = scratchBufferName(scratchSlot);
            String type = renderDataTypeToken(scratchBuffer.dataType());
            registerBuffer(scratchBuffer, name);
            addLine(
                    "__global "
                            + type
                            + " *"
                            + name
                            + " = (__global "
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
            return "get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0";
        }

        @Override
        protected LinearIdSpec linearIdSpec() {
            return new LinearIdSpec("long", "(long)get_global_id(0)", "1L");
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
            String ptr = byteOffsetPointer("__global uchar *", buffer.name, offset);
            return derefRead("__global " + renderDataTypeToken(type) + " *", ptr);
        }

        @Override
        protected String renderBufferWriteStatement(
                BufferBinding buffer,
                DataType type,
                String offset,
                String value,
                DataType valueType) {
            String ptr = byteOffsetPointer("__global uchar *", buffer.name, offset);
            String storeValue = storeValueExpr(type, value, valueType);
            return derefWrite("__global " + renderDataTypeToken(type) + " *", ptr, storeValue);
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
            return CLikeKernelSignatureSupport.renderKernelArgumentListGrouped(
                    graph(),
                    this::renderKernelInputArgument,
                    this::renderKernelOutputArgument,
                    new String[] {"__global uchar *scratch"});
        }

        private String renderKernelInputArgument(LIRInput input, int argIndex) {
            if (input instanceof BufferRef buffer) {
                return "__global const "
                        + renderDataTypeToken(buffer.dataType())
                        + " *input"
                        + argIndex;
            }
            ScalarInput scalar = requireScalarInput(input);
            return renderDataTypeToken(scalar.dataType()) + " scalar" + argIndex;
        }

        private String renderKernelOutputArgument(BufferRef output, int outputIndex, int ignored) {
            return "__global " + renderDataTypeToken(output.dataType()) + " *output" + outputIndex;
        }
    }
}
