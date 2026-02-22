package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TextRenderUtils;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Renders IR-T (Tensor IR) graphs as MLIR-style text representation.
 *
 * <p>Output format characteristics:
 *
 * <ul>
 *   <li>SSA-style variable naming with %0, %1, etc.
 *   <li>Explicit data types on all operations (FP32, I32, etc.)
 *   <li>Full layout information (shape:stride format)
 *   <li>Compact operation syntax: {@code %2 = add FP32 %0, %1}
 *   <li>Structured representation with inputs, outputs, and body sections
 * </ul>
 */
public class TIRTextRenderer implements TIRVisitor<String> {

    private final StringBuilder output;
    private final int indentSize;
    private int currentIndent;
    private int nextVarId;
    private int nextInputId;
    private int nextOutputId;
    private final Map<TIRNode, String> nodeToVar;
    private boolean inGraphRenderMode;
    private final boolean showViews;

    /** Creates a new renderer with default settings. */
    public TIRTextRenderer() {
        this(2, false);
    }

    /** Creates a new renderer with specified indent size. */
    public TIRTextRenderer(int indentSize) {
        this(indentSize, false);
    }

    /** Creates a new renderer with specified indent size and view rendering option. */
    public TIRTextRenderer(int indentSize, boolean showViews) {
        this.output = new StringBuilder();
        this.indentSize = indentSize;
        this.currentIndent = 0;
        this.nextVarId = 0;
        this.nodeToVar = new IdentityHashMap<>();
        this.inGraphRenderMode = false;
        this.showViews = showViews;
    }

    /** Renders a TIRGraph to MLIR-style text. */
    public String render(TIRGraph graph) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nextInputId = 0;
        nextOutputId = 0;
        nodeToVar.clear();
        inGraphRenderMode = true;

        // Build parameter list (inputs + outputs)
        StringBuilder params = new StringBuilder();
        for (int i = 0; i < graph.inputs().size(); i++) {
            TIRNode input = graph.inputs().get(i);
            if (!nodeToVar.containsKey(input)) {
                allocateInputVar(input);
            }
            String var = nodeToVar.get(input);
            params.append(var).append(": ").append(formatInput(input, i));
            if (i < graph.inputs().size() - 1 || !graph.outputs().isEmpty()) {
                params.append(", ");
            }
        }
        for (int i = 0; i < graph.outputs().size(); i++) {
            TIRNode outputNode = graph.outputs().get(i);
            if (!nodeToVar.containsKey(outputNode)) {
                allocateOutputVar(outputNode);
            }
            String var = nodeToVar.get(outputNode);
            params.append(var).append(": ").append(formatOutput(outputNode, i));
            if (i < graph.outputs().size() - 1) {
                params.append(", ");
            }
        }

        appendLine("module {");
        increaseIndent();
        appendLine("func.func @tensor_ops(" + params + ") {");
        increaseIndent();
        for (TIRNode outputNode : graph.outputs()) {
            renderNodeForGraph(outputNode);
        }
        decreaseIndent();
        appendLine("}");
        decreaseIndent();
        appendLine("}");

        inGraphRenderMode = false;
        return output.toString();
    }

    private void renderNodeForGraph(TIRNode node) {
        if (nodeToVar.containsKey(node) && node instanceof TensorInput) {
            return;
        }

        if (!nodeToVar.containsKey(node)) {
            allocateVar(node);
        }

        if (node instanceof TensorInput) {
            renderInputNodeForGraph((TensorInput) node);
        } else if (node instanceof ScalarConstant) {
            renderScalarConstantForGraph((ScalarConstant) node);
        } else if (node instanceof IotaConstant) {
            renderIotaConstantForGraph((IotaConstant) node);
        } else if (node instanceof UnaryOp) {
            renderUnaryOpForGraph((UnaryOp) node);
        } else if (node instanceof BinaryOp) {
            renderBinaryOpForGraph((BinaryOp) node);
        } else if (node instanceof TernaryOp) {
            renderTernaryOpForGraph((TernaryOp) node);
        } else if (node instanceof CastOp) {
            renderCastOpForGraph((CastOp) node);
        } else if (node instanceof ReductionOp) {
            renderReductionOpForGraph((ReductionOp) node);
        } else if (node instanceof ViewTransform) {
            renderViewTransformForGraph((ViewTransform) node);
        } else if (node instanceof Contiguous) {
            renderContiguousForGraph((Contiguous) node);
        }
    }

    private void renderInputNodeForGraph(TensorInput node) {}

    private void renderScalarConstantForGraph(ScalarConstant node) {
        String var = nodeToVar.get(node);
        String value = TextRenderUtils.formatScalarValue(node.rawBits(), node.dataType());
        String scalarDef =
                var
                        + " = arith.constant "
                        + value
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(scalarDef);
    }

    private void renderIotaConstantForGraph(IotaConstant node) {
        String var = nodeToVar.get(node);
        String iotaDef =
                var
                        + " = tensor.generate(%i) { arith.index_cast %i : index to "
                        + formatDataType(node.dataType())
                        + " } : tensor<"
                        + node.count()
                        + "x"
                        + formatDataType(node.dataType())
                        + ">";
        appendLine(iotaDef);
    }

    private void renderUnaryOpForGraph(UnaryOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String op = formatUnaryOp(node.op());
        String var = nodeToVar.get(node);
        String unaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(unaryDef);
    }

    private void renderBinaryOpForGraph(BinaryOp node) {
        if (!nodeToVar.containsKey(node.left())) {
            renderNodeForGraph(node.left());
        }
        if (!nodeToVar.containsKey(node.right())) {
            renderNodeForGraph(node.right());
        }
        String left = nodeToVar.get(node.left());
        String right = nodeToVar.get(node.right());
        String op = formatBinaryOp(node.op());
        String var = nodeToVar.get(node);
        String binaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + left
                        + ", "
                        + right
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(binaryDef);
    }

    private void renderTernaryOpForGraph(TernaryOp node) {
        if (!nodeToVar.containsKey(node.cond())) {
            renderNodeForGraph(node.cond());
        }
        if (!nodeToVar.containsKey(node.trueExpr())) {
            renderNodeForGraph(node.trueExpr());
        }
        if (!nodeToVar.containsKey(node.falseExpr())) {
            renderNodeForGraph(node.falseExpr());
        }
        String cond = nodeToVar.get(node.cond());
        String trueExpr = nodeToVar.get(node.trueExpr());
        String falseExpr = nodeToVar.get(node.falseExpr());
        String var = nodeToVar.get(node);
        String ternaryDef =
                var
                        + " = arith.select "
                        + cond
                        + ", "
                        + trueExpr
                        + ", "
                        + falseExpr
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(ternaryDef);
    }

    private void renderCastOpForGraph(CastOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        String op = formatCastOp(node.input().dataType(), node.targetDataType());
        String castDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(
                                node.targetDataType());
        appendLine(castDef);
    }

    private void renderReductionOpForGraph(ReductionOp node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        String input = nodeToVar.get(node.input());
        String op = formatReductionOp(node.op());
        String var = nodeToVar.get(node);
        StringBuilder axesStr = new StringBuilder();
        for (int i = 0; i < node.axes().length; i++) {
            if (i > 0) {
                axesStr.append(", ");
            }
            axesStr.append(node.axes()[i]);
        }
        String reductionDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + " axes=["
                        + axesStr
                        + "] keepDims="
                        + node.keepDims();
        appendLine(reductionDef);
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

    private String formatReductionOp(ReductionOperator op) {
        return "tensor." + op.name().toLowerCase();
    }

    private void renderViewTransformForGraph(ViewTransform node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        if (!showViews) {
            if (!nodeToVar.containsKey(node)) {
                nodeToVar.put(node, nodeToVar.get(node.input()));
            }
            return;
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        appendLine(formatViewTransformOp(node, input, var));
    }

    private void renderContiguousForGraph(Contiguous node) {
        if (!nodeToVar.containsKey(node.input())) {
            renderNodeForGraph(node.input());
        }
        if (!showViews) {
            if (!nodeToVar.containsKey(node)) {
                nodeToVar.put(node, nodeToVar.get(node.input()));
            }
            return;
        }
        String input = nodeToVar.get(node.input());
        String var = nodeToVar.get(node);
        appendLine(formatContiguousOp(node, input, var));
    }

    /** Renders a single TIRNode to a string. */
    public String renderNode(TIRNode node) {
        output.setLength(0);
        currentIndent = 0;
        nextVarId = 0;
        nodeToVar.clear();
        inGraphRenderMode = false;
        node.accept(this);
        return output.toString().trim();
    }

    private String formatInput(TIRNode node, int id) {
        if (node instanceof TensorInput input) {
            long[] shape = input.layout().shape().toArray();
            long[] stride = input.layout().stride().toArray();
            boolean contiguous = com.qxotic.jota.ir.TextRenderUtils.isContiguous(shape, stride);
            return com.qxotic.jota.ir.TextRenderUtils.formatTensorType(
                    input.dataType(), shape, stride, contiguous);
        }
        return formatNodeType(node);
    }

    private String formatOutput(TIRNode node, int id) {
        return formatTensorTypeForNode(node);
    }

    private String formatNodeType(TIRNode node) {
        return formatTensorTypeForNode(node);
    }

    private String formatTensorType(Layout layout, DataType dataType) {
        long[] shape = layout.shape().toArray();
        long[] stride = layout.stride().toArray();
        boolean contiguous = com.qxotic.jota.ir.TextRenderUtils.isContiguous(shape, stride);
        return com.qxotic.jota.ir.TextRenderUtils.formatTensorType(
                dataType, shape, stride, contiguous);
    }

    private String formatTensorTypeForNode(TIRNode node) {
        if (node instanceof TensorInput input) {
            return formatTensorType(input.layout(), input.dataType());
        }
        if (node instanceof ViewTransform view) {
            return formatTensorType(view.layout(), view.dataType());
        }
        long[] shape = node.shape().toArray();
        return com.qxotic.jota.ir.TextRenderUtils.formatTensorType(
                node.dataType(), shape, null, true);
    }

    private Layout layoutForNode(TIRNode node) {
        if (node instanceof TensorInput input) {
            return input.layout();
        }
        if (node instanceof ViewTransform view) {
            return view.layout();
        }
        return Layout.rowMajor(node.shape());
    }

    private String formatViewTransformOp(ViewTransform node, String input, String var) {
        String inputType = formatTensorType(layoutForNode(node.input()), node.dataType());
        String outputType = formatTensorType(node.layout(), node.dataType());
        String op =
                switch (node.kind()) {
                    case ViewKind.Transpose t ->
                            "tensor.transpose "
                                    + input
                                    + " perm=["
                                    + formatIntList(t.permutation())
                                    + "]";
                    case ViewKind.Reshape __ -> "tensor.reshape " + input;
                    case ViewKind.Broadcast __ -> "tensor.broadcast " + input;
                    case ViewKind.Expand __ -> "tensor.expand " + input;
                    case ViewKind.Slice s ->
                            "tensor.slice "
                                    + input
                                    + " axis="
                                    + s.axis()
                                    + " start="
                                    + s.start()
                                    + " step="
                                    + s.step();
                };
        return var
                + " = "
                + op
                + " : "
                + inputType
                + " to "
                + outputType
                + "  // layout=in"
                + layoutForNode(node.input())
                + ", out"
                + node.layout();
    }

    private String formatContiguousOp(Contiguous node, String input, String var) {
        String inputType = formatTensorType(layoutForNode(node.input()), node.dataType());
        String outputType = formatTensorType(Layout.rowMajor(node.shape()), node.dataType());
        return var
                + " = tensor.contiguous "
                + input
                + " : "
                + inputType
                + " to "
                + outputType
                + "  // layout=in"
                + layoutForNode(node.input())
                + ", out"
                + Layout.rowMajor(node.shape());
    }

    private String formatIntList(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values[i]);
        }
        return sb.toString();
    }

    private String formatDataType(DataType dataType) {
        return com.qxotic.jota.ir.TextRenderUtils.formatDataType(dataType);
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

    private String allocateVar(TIRNode node) {
        String varName = "%" + nextVarId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String allocateInputVar(TIRNode node) {
        String varName = "%in" + nextInputId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String allocateOutputVar(TIRNode node) {
        String varName = "%out" + nextOutputId++;
        nodeToVar.put(node, varName);
        return varName;
    }

    private String getVar(TIRNode node) {
        if (nodeToVar.containsKey(node)) {
            return nodeToVar.get(node);
        }
        String result = node.accept(this);
        return result;
    }

    @Override
    public String visitTensorInput(TensorInput node) {
        String var = allocateVar(node);
        return var;
    }

    @Override
    public String visitScalarInput(ScalarInput node) {
        String var = allocateVar(node);
        return var;
    }

    @Override
    public String visitScalarConstant(ScalarConstant node) {
        String var = allocateVar(node);
        String value = TextRenderUtils.formatScalarValue(node.rawBits(), node.dataType());
        String scalarDef =
                var
                        + " = arith.constant "
                        + value
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(scalarDef);
        return var;
    }

    @Override
    public String visitIotaConstant(IotaConstant node) {
        String var = allocateVar(node);
        String iotaDef =
                var
                        + " = tensor.generate(%i) { arith.index_cast %i : index to "
                        + TextRenderUtils.formatDataType(node.dataType())
                        + " } : tensor<"
                        + node.count()
                        + "x"
                        + TextRenderUtils.formatDataType(node.dataType())
                        + ">";
        appendLine(iotaDef);
        return var;
    }

    @Override
    public String visitUnaryOp(UnaryOp node) {
        String input = node.input().accept(this);
        String op = formatUnaryOp(node.op());
        String var = allocateVar(node);
        String unaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(unaryDef);
        return var;
    }

    @Override
    public String visitBinaryOp(BinaryOp node) {
        String left = node.left().accept(this);
        String right = node.right().accept(this);
        String op = formatBinaryOp(node.op());
        String var = allocateVar(node);
        String binaryDef =
                var
                        + " = "
                        + op
                        + " "
                        + left
                        + ", "
                        + right
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(binaryDef);
        return var;
    }

    @Override
    public String visitTernaryOp(TernaryOp node) {
        String cond = node.cond().accept(this);
        String trueExpr = node.trueExpr().accept(this);
        String falseExpr = node.falseExpr().accept(this);
        String var = allocateVar(node);
        String ternaryDef =
                var
                        + " = arith.select "
                        + cond
                        + ", "
                        + trueExpr
                        + ", "
                        + falseExpr
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(node.dataType());
        appendLine(ternaryDef);
        return var;
    }

    @Override
    public String visitCastOp(CastOp node) {
        String input = node.input().accept(this);
        String op = formatCastOp(node.input().dataType(), node.targetDataType());
        String var = allocateVar(node);
        String castDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + com.qxotic.jota.ir.TextRenderUtils.formatOpTypeSuffix(
                                node.targetDataType());
        appendLine(castDef);
        return var;
    }

    @Override
    public String visitReductionOp(ReductionOp node) {
        String input = node.input().accept(this);
        String op = formatReductionOp(node.op());
        String var = allocateVar(node);
        StringBuilder axesStr = new StringBuilder();
        for (int i = 0; i < node.axes().length; i++) {
            if (i > 0) {
                axesStr.append(", ");
            }
            axesStr.append(node.axes()[i]);
        }
        String reductionDef =
                var
                        + " = "
                        + op
                        + " "
                        + input
                        + " dimensions=["
                        + axesStr
                        + "] keepDims="
                        + node.keepDims();
        appendLine(reductionDef);
        return var;
    }

    @Override
    public String visitGatherOp(GatherOp node) {
        String input = node.input().accept(this);
        String indices = node.indices().accept(this);
        String var = allocateVar(node);
        String gatherDef =
                var
                        + " = gather "
                        + input
                        + " indices="
                        + indices
                        + " axis="
                        + node.axis()
                        + " -> "
                        + node.shape();
        appendLine(gatherDef);
        return var;
    }

    @Override
    public String visitViewTransform(ViewTransform node) {
        String input = node.input().accept(this);
        String var = allocateVar(node);
        appendLine(formatViewTransformOp(node, input, var));
        return var;
    }

    @Override
    public String visitContiguous(Contiguous node) {
        String input = node.input().accept(this);
        String var = allocateVar(node);
        appendLine(formatContiguousOp(node, input, var));
        return var;
    }
}
