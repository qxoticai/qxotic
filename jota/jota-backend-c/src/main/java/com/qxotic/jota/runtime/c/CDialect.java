package com.qxotic.jota.runtime.c;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRExprNode;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.StructuredFor;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.runtime.clike.CLikeDataTypeModel;
import com.qxotic.jota.runtime.clike.CLikeDialect;
import com.qxotic.jota.runtime.clike.CLikeLirSourceGenerator;
import com.qxotic.jota.runtime.clike.CLikeParallelLoopSupport.LinearIdSpec;
import java.util.Map;

final class CDialect implements CLikeDialect {

    private static final CLikeDataTypeModel TYPE_MODEL =
            new CLikeDataTypeModel(
                    Map.of(
                            DataType.BOOL, "uint8_t",
                            DataType.I8, "int8_t",
                            DataType.I16, "int16_t",
                            DataType.I32, "int32_t",
                            DataType.I64, "int64_t",
                            DataType.FP16, "_Float16",
                            DataType.BF16, "__bf16",
                            DataType.FP32, "float",
                            DataType.FP64, "double"),
                    Map.of(
                            DataType.I8, "uint8_t",
                            DataType.I16, "uint16_t",
                            DataType.I32, "uint32_t",
                            DataType.I64, "uint64_t"));

    @Override
    public String language() {
        return "c";
    }

    @Override
    public String renderSource(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
        return new CEmitter(graph, scratchLayout, kernelName).generate();
    }

    private static final class CEmitter extends CLikeLirSourceGenerator {
        private CEmitter(LIRGraph graph, ScratchLayout scratchLayout, String kernelName) {
            super(graph, scratchLayout, kernelName);
        }

        @Override
        protected void appendPreamble(StringBuilder source) {
            source.append("#include <stdint.h>\n");
            source.append("#include <stddef.h>\n");
            source.append("#include <string.h>\n");
            source.append("#include <math.h>\n\n");
            source.append("// openmp: ").append(COpenMpConfig.enabled()).append("\n");
        }

        @Override
        protected String kernelDeclaration() {
            return "void "
                    + kernelName()
                    + "(void **buffers, uint64_t *scalars, uint64_t scratch_ptr)";
        }

        @Override
        protected void emitProlog() {
            int bufferIndex = 0;
            int scalarIndex = 0;
            for (LIRInput input : graph().inputs()) {
                if (input instanceof BufferRef buffer) {
                    String name = inputBufferName(bufferIndex);
                    addLine("uint8_t *" + name + " = (uint8_t *)buffers[" + bufferIndex + "];");
                    bindReadOnlyInputBuffer(buffer, bufferIndex);
                    bufferIndex++;
                } else {
                    ScalarInput scalar = requireScalarInput(input);
                    String name = scalarLocalName(scalarIndex);
                    registerScalarInputName(scalar.id(), name);
                    String bitsExpr = "scalars[" + scalarIndex + "]";
                    emitScalarUnpack(name, scalar.dataType(), bitsExpr);
                    scalarIndex++;
                }
            }

            for (int i = 0; i < graph().outputs().size(); i++) {
                BufferRef buffer = graph().outputs().get(i);
                String name = outputBufferName(i);
                addLine("uint8_t *" + name + " = (uint8_t *)buffers[" + (bufferIndex + i) + "];");
                registerBuffer(buffer, name);
            }

            addLine("uint8_t *scratch = (uint8_t *)(uintptr_t)scratch_ptr;");
            emitScratchBindings(this::bindScratch);
        }

        private void bindScratch(BufferRef scratchBuffer, long offset, int scratchSlot) {
            String name = scratchBufferName(scratchSlot);
            registerBuffer(scratchBuffer, name);
            addLine("uint8_t *" + name + " = scratch + " + offset + "LL;");
        }

        @Override
        protected void maybeEmitSimpleLoopPragma(StructuredFor loop) {
            if (COpenMpConfig.enabled() && loop.iterArgs().isEmpty()) {
                addLine("#pragma omp parallel for");
            }
        }

        @Override
        protected boolean tryEmitParallelTopLevel(LIRExprNode body) {
            return false;
        }

        @Override
        protected LinearIdSpec linearIdSpec() {
            return new LinearIdSpec("long long", "0", "1LL");
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
            String ptr = byteOffsetPointer("uint8_t *", buffer.name, offset);
            return derefRead(renderDataTypeToken(type) + " *", ptr);
        }

        @Override
        protected String renderBufferWriteStatement(
                BufferBinding buffer,
                DataType type,
                String offset,
                String value,
                DataType valueType) {
            String ptr = byteOffsetPointer("uint8_t *", buffer.name, offset);
            String storeValue = storeValueExpr(type, value, valueType);
            return derefWrite(renderDataTypeToken(type) + " *", ptr, storeValue);
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

        @Override
        protected boolean shouldAvoidTempNameCollisionWithScalars() {
            return false;
        }

        private void emitScalarUnpack(String name, DataType type, String bitsExpr) {
            if (type == DataType.FP16) {
                emitBitcastScalarUnpack(name, "_Float16", "uint16_t", bitsExpr, "uint16_t");
            } else if (type == DataType.BF16) {
                emitBitcastScalarUnpack(name, "__bf16", "uint16_t", bitsExpr, "uint16_t");
            } else if (type == DataType.FP32) {
                emitBitcastScalarUnpack(name, "float", "uint32_t", bitsExpr, "float");
            } else if (type == DataType.FP64) {
                emitBitcastScalarUnpack(name, "double", "uint64_t", bitsExpr, "double");
            } else {
                addLine(
                        renderDataTypeToken(type)
                                + " "
                                + name
                                + " = ("
                                + renderDataTypeToken(type)
                                + ")"
                                + bitsExpr
                                + ";");
            }
        }

        private void emitBitcastScalarUnpack(
                String name,
                String targetType,
                String bitsType,
                String bitsExpr,
                String memcpySizeType) {
            addLine(bitsType + " " + name + "_bits = (" + bitsType + ")" + bitsExpr + ";");
            addLine(
                    targetType
                            + " "
                            + name
                            + "; memcpy(&"
                            + name
                            + ", &"
                            + name
                            + "_bits, sizeof("
                            + memcpySizeType
                            + "));");
        }
    }
}
