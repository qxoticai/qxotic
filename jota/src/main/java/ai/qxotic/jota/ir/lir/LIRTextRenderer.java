package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.TextRenderUtils;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.util.HashMap;
import java.util.Map;

/**
 * Renders IR-L graphs as MLIR-style text representation.
 *
 * <p>Output format characteristics:
 *
 * <ul>
 *   <li>SSA-style variable naming with descriptive buffer names
 *   <li>Explicit data types on all operations
 *   <li>Inline index expressions
 *   <li>Full buffer metadata (shape, strides)
 *   <li>Structured loop notation: {@code for %i = 0 to 4 step 1}
 *   <li>Flat representation with intermediate variables
 * </ul>
 */
public class LIRTextRenderer implements LIRVisitor<String> {

    private final StringBuilder output;
    private final int indentSize;
    private int currentIndent;
    private int nextVarId;
    private final Map<ScalarExpr, String> exprToVar;
    private final Map<Integer, String> bufferIdToName;
    private int nextInputId;
    private int nextOutputId;

    /** Creates a new renderer with default settings. */
    public LIRTextRenderer() {
        this(2);
    }

    /** Creates a new renderer with specified indent size. */
    public LIRTextRenderer(int indentSize) {
        this.output = new StringBuilder();
        this.indentSize = indentSize;
        this.currentIndent = 0;
        this.nextVarId = 0;
        this.exprToVar = new HashMap<>();
        this.bufferIdToName = new HashMap<>();
        this.nextInputId = 0;
        this.nextOutputId = 0;
    }

    /** Renders an LIRGraph to MLIR-style text. */
    public String render(LIRGraph graph) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nextInputId = 0;
        nextOutputId = 0;
        exprToVar.clear();
        bufferIdToName.clear();

        // Build parameter list (inputs + outputs)
        StringBuilder params = new StringBuilder();
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            String name = "in" + nextInputId++;
            bufferIdToName.put(input.id(), name);

            switch (input) {
                case BufferRef buf ->
                        params.append("%")
                                .append(name)
                                .append(": ")
                                .append(formatMemRefType(buf.dataType(), buf.layout()));
                case ScalarInput scalar ->
                        params.append("%")
                                .append(name)
                                .append(": ")
                                .append(TextRenderUtils.formatDataType(scalar.dataType()));
            }
            if (i < graph.inputs().size() - 1 || !graph.outputs().isEmpty()) {
                params.append(", ");
            }
        }
        for (int i = 0; i < graph.outputs().size(); i++) {
            BufferRef outputBuf = graph.outputs().get(i);
            String name = "out" + nextOutputId++;
            bufferIdToName.put(outputBuf.id(), name);

            params.append("%")
                    .append(name)
                    .append(": ")
                    .append(formatMemRefType(outputBuf.dataType(), outputBuf.layout()));
            if (i < graph.outputs().size() - 1) {
                params.append(", ");
            }
        }

        appendLine("module {");
        increaseIndent();
        appendLine("func.func @kernel(" + params + ") {");
        increaseIndent();

        graph.body().accept(this);

        decreaseIndent();
        appendLine("  return");
        appendLine("}");
        decreaseIndent();
        appendLine("}");

        return output.toString();
    }

    private String formatMemRefType(DataType dataType, Layout layout) {
        int rank = (int) layout.shape().flatRank();
        long[] shape = new long[rank];
        long[] strides = new long[rank];
        long byteSize = dataType.byteSize();
        for (int i = 0; i < rank; i++) {
            shape[i] = layout.shape().flatAt(i);
            strides[i] = layout.stride().flatAt(i) * byteSize;
        }
        boolean contiguous =
                TextRenderUtils.isContiguous(shape, strides, (int) dataType.byteSize());
        return TextRenderUtils.formatMemRefType(
                dataType, shape, strides, (int) byteSize, contiguous);
    }

    private void appendLine(String line) {
        appendIndent();
        output.append(line);
        output.append("\n");
    }

    private void append(String text) {
        output.append(text);
    }

    private void appendIndent() {
        for (int i = 0; i < currentIndent * indentSize; i++) {
            output.append(" ");
        }
    }

    private void increaseIndent() {
        currentIndent++;
    }

    private void decreaseIndent() {
        currentIndent--;
    }

    private String allocateVar(ScalarExpr expr, String rendered) {
        String varName = "%" + nextVarId++;
        exprToVar.put(expr, varName);
        return varName + " = " + rendered;
    }

    private String getVar(ScalarExpr expr) {
        if (exprToVar.containsKey(expr)) {
            return exprToVar.get(expr);
        }
        return expr.accept(this);
    }

    /**
     * Renders the RHS of an expression (the operation without variable assignment). Used for
     * ScalarLet to avoid creating intermediate variables.
     */
    private String renderExprRhs(ScalarExpr expr) {
        return switch (expr) {
            case ScalarLiteral lit -> visitScalarLiteral(lit);
            case ScalarInput input -> visitScalarInput(input);
            case ScalarRef ref -> visitScalarRef(ref);
            case ScalarFromIndex sfi ->
                    "arith.index_cast "
                            + sfi.index().accept(this)
                            + TextRenderUtils.formatOpTypeSuffix(sfi.dataType());
            case ScalarLoad load ->
                    "memref.load %"
                            + bufferName(load.buffer())
                            + "["
                            + load.offset().accept(this)
                            + "] : "
                            + formatMemRefType(load.dataType(), load.buffer().layout());
            case ScalarCast cast ->
                    formatCastOp(cast.input().dataType(), cast.targetType())
                            + " "
                            + getVar(cast.input())
                            + TextRenderUtils.formatOpTypeSuffix(cast.targetType());
            case ScalarUnary unary ->
                    formatUnaryOp(unary.op())
                            + " "
                            + getVar(unary.input())
                            + TextRenderUtils.formatOpTypeSuffix(unary.dataType());
            case ScalarBinary binary ->
                    formatBinaryOp(binary.op())
                            + " "
                            + getVar(binary.left())
                            + ", "
                            + getVar(binary.right())
                            + TextRenderUtils.formatOpTypeSuffix(binary.dataType());
            case ScalarTernary ternary ->
                    "arith.select "
                            + ternary.dataType()
                            + " "
                            + getVar(ternary.condition())
                            + ", "
                            + getVar(ternary.trueValue())
                            + ", "
                            + getVar(ternary.falseValue());
        };
    }

    // ==================== Index Expressions ====================

    @Override
    public String visitIndexVar(IndexVar node) {
        return "%" + node.name();
    }

    @Override
    public String visitIndexConst(IndexConst node) {
        return String.valueOf(node.value());
    }

    @Override
    public String visitIndexBinary(IndexBinary node) {
        String left = node.left().accept(this);
        String right = node.right().accept(this);
        String op = formatIndexBinaryOp(node.op());

        // Always wrap IndexBinary children in parentheses to show exact tree structure
        if (node.left() instanceof IndexBinary) {
            left = "(" + left + ")";
        }
        if (node.right() instanceof IndexBinary) {
            right = "(" + right + ")";
        }

        return left + " " + op + " " + right;
    }

    private String formatIndexBinaryOp(IndexBinary.IndexBinaryOp op) {
        return switch (op) {
            case ADD -> "+";
            case SUBTRACT -> "-";
            case MULTIPLY -> "*";
            case DIVIDE -> "/";
            case MODULO -> "%";
            case BITWISE_AND -> "&";
            case SHIFT_LEFT -> "<<";
            case SHIFT_RIGHT -> ">>";
        };
    }

    // ==================== Scalar Expressions ====================

    @Override
    public String visitScalarLiteral(ScalarLiteral node) {
        DataType dt = node.dataType();
        if (dt == DataType.FP32) {
            return node.asFloat() + "f";
        } else if (dt == DataType.FP64) {
            return node.asDouble() + "";
        } else if (dt == DataType.I32) {
            return String.valueOf(node.asInt());
        } else if (dt == DataType.I64) {
            return String.valueOf(node.asLong());
        } else if (dt == DataType.BOOL) {
            return node.asBool() ? "true" : "false";
        } else if (dt == DataType.FP16) {
            return TextRenderUtils.formatFloat16((short) node.rawBits());
        } else if (dt == DataType.BF16) {
            return TextRenderUtils.formatBFloat16((short) node.rawBits());
        } else {
            return "0x" + Long.toHexString(node.rawBits());
        }
    }

    @Override
    public String visitScalarUnary(ScalarUnary node) {
        String input = getVar(node.input());
        String op = formatUnaryOp(node.op());
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(
                varName
                        + " = "
                        + op
                        + " "
                        + input
                        + TextRenderUtils.formatOpTypeSuffix(dt)
                        + "  // "
                        + getUnaryOpComment(node.op()));
        return varName;
    }

    @Override
    public String visitScalarBinary(ScalarBinary node) {
        String left = getVar(node.left());
        String right = getVar(node.right());
        String op = formatBinaryOp(node.op());
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(
                varName
                        + " = "
                        + op
                        + " "
                        + left
                        + ", "
                        + right
                        + TextRenderUtils.formatOpTypeSuffix(dt)
                        + "  // "
                        + getBinaryOpComment(node.op(), left, right));
        return varName;
    }

    @Override
    public String visitScalarTernary(ScalarTernary node) {
        String cond = getVar(node.condition());
        String trueVal = getVar(node.trueValue());
        String falseVal = getVar(node.falseValue());
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(
                varName
                        + " = arith.select "
                        + cond
                        + ", "
                        + trueVal
                        + ", "
                        + falseVal
                        + TextRenderUtils.formatOpTypeSuffix(dt));
        return varName;
    }

    @Override
    public String visitScalarCast(ScalarCast node) {
        String input = getVar(node.input());
        DataType target = node.targetType();
        DataType source = node.input().dataType();
        String op = formatCastOp(source, target);
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(varName + " = " + op + " " + input + TextRenderUtils.formatOpTypeSuffix(target));
        return varName;
    }

    @Override
    public String visitScalarLoad(ScalarLoad node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(
                varName
                        + " = memref.load %"
                        + bufferName(buffer)
                        + "["
                        + offset
                        + "] : "
                        + formatMemRefType(dt, buffer.layout())
                        + "  // input");
        return varName;
    }

    @Override
    public String visitScalarInput(ScalarInput node) {
        String name = bufferIdToName.get(node.id());
        if (name == null) {
            name = "in" + node.id();
        }
        return "%" + name;
    }

    @Override
    public String visitScalarFromIndex(ScalarFromIndex node) {
        String index = node.index().accept(this);
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        exprToVar.put(node, varName);
        appendLine(
                varName + " = arith.index_cast " + index + TextRenderUtils.formatOpTypeSuffix(dt));
        return varName;
    }

    @Override
    public String visitScalarRef(ScalarRef node) {
        return "%" + node.name();
    }

    @Override
    public String visitScalarLet(ScalarLet node) {
        String rhs = renderExprRhs(node.value());
        String varName = "%" + node.name();
        exprToVar.put(node.value(), varName);
        appendLine(varName + " = " + rhs);
        return "";
    }

    private String formatUnaryOp(UnaryOperator op) {
        return switch (op) {
            case NEGATE -> "arith.negf";
            case ABS -> "math.abs";
            case EXP -> "math.exp";
            case LOG -> "math.log";
            case SQRT -> "math.sqrt";
            case TANH -> "math.tanh";
            case RECIPROCAL -> "arith.divf 1.0";
            default -> "arith." + op.name().toLowerCase();
        };
    }

    private String formatBinaryOp(BinaryOperator op) {
        return switch (op) {
            case ADD -> "arith.addf";
            case SUBTRACT -> "arith.subf";
            case MULTIPLY -> "arith.mulf";
            case DIVIDE -> "arith.divf";
            case MIN -> "arith.minf";
            case MAX -> "arith.maxf";
            case POW -> "math.powf";
            default -> "arith." + op.name().toLowerCase();
        };
    }

    private String formatCastOp(DataType source, DataType target) {
        if (source.isFloatingPoint() && target.isIntegral()) {
            return "arith.fptosi";
        } else if (source.isIntegral() && target.isFloatingPoint()) {
            return "arith.sitofp";
        } else if (source.byteSize() == target.byteSize()) {
            return "arith.bitcast";
        } else {
            return "arith.extf";
        }
    }

    private String getUnaryOpComment(UnaryOperator op) {
        return op.name().toLowerCase();
    }

    private String getBinaryOpComment(BinaryOperator op, String left, String right) {
        if (left.equals(right)) {
            return switch (op) {
                case MULTIPLY -> "squared";
                case ADD -> "doubled";
                default -> op.name().toLowerCase();
            };
        }
        return op.name().toLowerCase();
    }

    private String bufferName(BufferRef buffer) {
        // Look up the buffer name from the map, or use a generic name
        return bufferIdToName.getOrDefault(buffer.id(), "buffer" + buffer.id());
    }

    // ==================== Memory Operations ====================

    @Override
    public String visitBufferRef(BufferRef node) {
        return "%" + bufferName(node);
    }

    @Override
    public String visitLoad(Load node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        DataType dt = node.dataType();
        String varName = "%" + nextVarId++;
        appendLine(
                varName
                        + " = memref.load %"
                        + bufferName(buffer)
                        + "["
                        + offset
                        + "] : "
                        + formatMemRefType(dt, buffer.layout()));
        return varName;
    }

    @Override
    public String visitStore(Store node) {
        String offset = node.offset().accept(this);
        BufferRef buffer = node.buffer();
        String value = getVar(node.value());
        appendLine(
                "memref.store "
                        + value
                        + ", %"
                        + bufferName(buffer)
                        + "["
                        + offset
                        + "] : "
                        + formatMemRefType(node.value().dataType(), buffer.layout()));
        return "";
    }

    // ==================== Control Flow ====================

    @Override
    public String visitLoop(Loop node) {
        String bound = node.bound().accept(this);
        String loopType =
                node.isParallel()
                        ? "scf.parallel (%i) = (0) to (%b) step (1)"
                        : "scf.for %i = 0 to %b step 1";
        loopType = loopType.replace("%i", "%" + node.indexName()).replace("%b", bound);
        appendLine(loopType + " {");
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitStructuredFor(StructuredFor node) {
        String lb = node.lowerBound().accept(this);
        String ub = node.upperBound().accept(this);
        String step = node.step().accept(this);

        StringBuilder header = new StringBuilder();
        header.append("scf.for %")
                .append(node.indexName())
                .append(" = ")
                .append(lb)
                .append(" to ")
                .append(ub)
                .append(" step ")
                .append(step);

        if (!node.iterArgs().isEmpty()) {
            header.append(" iter_args(");
            for (int i = 0; i < node.iterArgs().size(); i++) {
                LoopIterArg arg = node.iterArgs().get(i);
                if (i > 0) {
                    header.append(", ");
                }
                header.append("%")
                        .append(arg.name())
                        .append(" = ")
                        .append(renderExprRhs(arg.init()));
            }
            header.append(") -> (");
            for (int i = 0; i < node.iterArgs().size(); i++) {
                if (i > 0) {
                    header.append(", ");
                }
                header.append(TextRenderUtils.formatDataType(node.iterArgs().get(i).dataType()));
            }
            header.append(")");
        }

        header.append(" {");
        appendLine(header.toString());
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitTiledLoop(TiledLoop node) {
        String totalBound = node.totalBound().accept(this);
        appendLine(
                "scf.for %"
                        + node.outerName()
                        + " = 0 to "
                        + totalBound
                        + " step "
                        + node.tileSize()
                        + " {");
        increaseIndent();
        appendLine(
                "  scf.for %"
                        + node.innerName()
                        + " = 0 to min(%"
                        + node.tileSize()
                        + ", "
                        + totalBound
                        + " - %"
                        + node.outerName()
                        + " * "
                        + node.tileSize()
                        + ") {");
        increaseIndent();
        node.body().accept(this);
        decreaseIndent();
        appendLine("  }");
        decreaseIndent();
        appendLine("}");
        return "";
    }

    @Override
    public String visitLoopNest(LoopNest node) {
        appendLine("// loop nest");
        increaseIndent();
        for (Loop loop : node.loops()) {
            String bound = loop.bound().accept(this);
            String loopType =
                    loop.isParallel()
                            ? "scf.parallel (%i) = (0) to (%b) step (1)"
                            : "scf.for %i = 0 to %b step 1";
            loopType = loopType.replace("%i", "%" + loop.indexName()).replace("%b", bound);
            appendLine(loopType);
        }
        decreaseIndent();
        node.body().accept(this);
        return "";
    }

    @Override
    public String visitBlock(Block node) {
        for (LIRNode statement : node.statements()) {
            statement.accept(this);
        }
        return "";
    }

    @Override
    public String visitYield(Yield node) {
        if (node.values().isEmpty()) {
            return "";
        }
        StringBuilder line = new StringBuilder();
        line.append("scf.yield ");
        for (int i = 0; i < node.values().size(); i++) {
            if (i > 0) {
                line.append(", ");
            }
            line.append(getVar(node.values().get(i)));
        }
        appendLine(line.toString());
        return "";
    }
}
