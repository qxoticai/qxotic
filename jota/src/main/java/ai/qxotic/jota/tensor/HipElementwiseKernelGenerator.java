package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class HipElementwiseKernelGenerator {

    public record GeneratedKernel(String name, String source) {}

    public GeneratedKernel generate(ExpressionGraph graph, String kernelName) {
        ExprNode root = graph.root();
        DataType outputType = root.dataType();
        String outputCType = typeName(outputType);

        List<InputNode> inputs = graph.inputs();
        StringBuilder signature = new StringBuilder();
        for (int i = 0; i < inputs.size(); i++) {
            InputNode input = inputs.get(i);
            String inType = typeName(input.dataType());
            signature.append("const ").append(inType).append(" *in").append(input.index());
            signature.append(", ");
        }
        signature.append(outputCType).append(" *out, int n");

        List<String> lines = new ArrayList<>();
        Map<ExprNode, String> temps = new HashMap<>();

        String rootExpr = buildExpr(root, temps, lines);

        StringBuilder body = new StringBuilder();
        body.append("#include <hip/hip_runtime.h>\n");
        body.append("#include <stdint.h>\n");
        body.append("#include <math.h>\n");
        body.append("extern \"C\" __global__ void ").append(kernelName).append("(")
                .append(signature).append(") {\n");
        body.append("  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
        body.append("  if (idx < n) {\n");
        for (String line : lines) {
            body.append("    ").append(line).append("\n");
        }
        body.append("    out[idx] = ").append(rootExpr).append(";\n");
        body.append("  }\n");
        body.append("}\n");

        return new GeneratedKernel(kernelName, body.toString());
    }

    private String buildExpr(ExprNode node, Map<ExprNode, String> temps, List<String> lines) {
        if (node instanceof InputNode input) {
            return "in" + input.index() + "[idx]";
        }
        if (node instanceof ScalarNode scalar) {
            return literal(scalar.value(), scalar.dataType());
        }
        if (node instanceof RangeNode range) {
            return "((" + typeName(range.dataType()) + ")idx)";
        }
        if (temps.containsKey(node)) {
            return temps.get(node);
        }

        String expr;
        DataType type = node.dataType();
        if (node instanceof CastNode cast) {
            String child = buildExpr(cast.input(), temps, lines);
            expr = castExpr(child, cast.targetType());
            type = cast.targetType();
        } else if (node instanceof UnaryNode unary) {
            String child = buildExpr(unary.input(), temps, lines);
            expr = unaryExpr(unary.op(), child, type);
        } else if (node instanceof BinaryNode binary) {
            String left = buildExpr(binary.left(), temps, lines);
            String right = buildExpr(binary.right(), temps, lines);
            expr = binaryExpr(binary.op(), left, right, type);
        } else if (node instanceof TernaryNode ternary) {
            String cond = buildExpr(ternary.condition(), temps, lines);
            String tVal = buildExpr(ternary.trueValue(), temps, lines);
            String fVal = buildExpr(ternary.falseValue(), temps, lines);
            expr = "((" + cond + " != 0) ? " + tVal + " : " + fVal + ")";
        } else {
            throw new UnsupportedOperationException("Unsupported HIP node: " + node.getClass());
        }

        String var = "t" + temps.size();
        String cType = typeName(type);
        lines.add(cType + " " + var + " = " + expr + ";");
        temps.put(node, var);
        return var;
    }

    private String castExpr(String expr, DataType targetType) {
        if (targetType == DataType.BOOL) {
            return "(" + expr + " != 0 ? 1 : 0)";
        }
        return "((" + typeName(targetType) + ")(" + expr + "))";
    }

    private String unaryExpr(UnaryOp op, String expr, DataType type) {
        if (op == UnaryOp.IDENTITY) {
            return expr;
        }
        if (op == UnaryOp.LOGICAL_NOT) {
            return "(" + expr + " ? 0 : 1)";
        }
        if (op == UnaryOp.BITWISE_NOT) {
            return "(~(" + expr + "))";
        }
        if (op == UnaryOp.NEGATE) {
            return "-" + expr;
        }
        if (op == UnaryOp.ABS) {
            if (type.isFloatingPoint()) {
                return type == DataType.FP64 ? "fabs(" + expr + ")" : "fabsf(" + expr + ")";
            }
            if (type.isIntegral()) {
                return "(" + expr + " < 0 ? -" + expr + " : " + expr + ")";
            }
            throw new UnsupportedOperationException("abs requires numeric type");
        }
        if (op == UnaryOp.SQUARE) {
            return "(" + expr + " * " + expr + ")";
        }
        if (!type.isFloatingPoint()) {
            throw new UnsupportedOperationException("Floating-point unary op requires FP32/FP64");
        }
        boolean fp64 = type == DataType.FP64;
        if (op == UnaryOp.EXP) {
            return fp64 ? "exp(" + expr + ")" : "expf(" + expr + ")";
        }
        if (op == UnaryOp.LOG) {
            return fp64 ? "log(" + expr + ")" : "logf(" + expr + ")";
        }
        if (op == UnaryOp.SQRT) {
            return fp64 ? "sqrt(" + expr + ")" : "sqrtf(" + expr + ")";
        }
        if (op == UnaryOp.SIN) {
            return fp64 ? "sin(" + expr + ")" : "sinf(" + expr + ")";
        }
        if (op == UnaryOp.COS) {
            return fp64 ? "cos(" + expr + ")" : "cosf(" + expr + ")";
        }
        if (op == UnaryOp.TANH) {
            return fp64 ? "tanh(" + expr + ")" : "tanhf(" + expr + ")";
        }
        if (op == UnaryOp.RECIPROCAL) {
            return fp64 ? "(1.0 / " + expr + ")" : "(1.0f / " + expr + ")";
        }
        throw new UnsupportedOperationException("Unsupported unary op: " + op);
    }

    private String binaryExpr(BinaryOp op, String left, String right, DataType type) {
        if (op == BinaryOp.EQUAL) {
            return "(" + left + " == " + right + " ? 1 : 0)";
        }
        if (op == BinaryOp.LESS_THAN) {
            return "(" + left + " < " + right + " ? 1 : 0)";
        }
        if (op == BinaryOp.LOGICAL_AND) {
            return "((" + left + " != 0) && (" + right + " != 0)) ? 1 : 0";
        }
        if (op == BinaryOp.LOGICAL_OR) {
            return "((" + left + " != 0) || (" + right + " != 0)) ? 1 : 0";
        }
        if (op == BinaryOp.LOGICAL_XOR) {
            return "((" + left + " != 0) ^ (" + right + " != 0)) ? 1 : 0";
        }
        if (op == BinaryOp.BITWISE_AND) {
            return "(" + left + " & " + right + ")";
        }
        if (op == BinaryOp.BITWISE_OR) {
            return "(" + left + " | " + right + ")";
        }
        if (op == BinaryOp.BITWISE_XOR) {
            return "(" + left + " ^ " + right + ")";
        }
        if (op == BinaryOp.MIN) {
            if (type.isFloatingPoint()) {
                return type == DataType.FP64
                        ? "fmin(" + left + ", " + right + ")"
                        : "fminf(" + left + ", " + right + ")";
            }
            return "(" + left + " < " + right + " ? " + left + " : " + right + ")";
        }
        if (op == BinaryOp.MAX) {
            if (type.isFloatingPoint()) {
                return type == DataType.FP64
                        ? "fmax(" + left + ", " + right + ")"
                        : "fmaxf(" + left + ", " + right + ")";
            }
            return "(" + left + " > " + right + " ? " + left + " : " + right + ")";
        }
        if (op == BinaryOp.POW) {
            if (!type.isFloatingPoint()) {
                throw new UnsupportedOperationException("pow requires floating-point type");
            }
            return type == DataType.FP64
                    ? "pow(" + left + ", " + right + ")"
                    : "powf(" + left + ", " + right + ")";
        }
        if (op == BinaryOp.ADD) {
            return "(" + left + " + " + right + ")";
        }
        if (op == BinaryOp.SUBTRACT) {
            return "(" + left + " - " + right + ")";
        }
        if (op == BinaryOp.MULTIPLY) {
            return "(" + left + " * " + right + ")";
        }
        if (op == BinaryOp.DIVIDE) {
            return "(" + left + " / " + right + ")";
        }
        throw new UnsupportedOperationException("Unsupported binary op: " + op);
    }

    private String typeName(DataType dataType) {
        if (dataType == DataType.BOOL) {
            return "uint8_t";
        }
        if (dataType == DataType.I8) {
            return "int8_t";
        }
        if (dataType == DataType.I16) {
            return "int16_t";
        }
        if (dataType == DataType.I32) {
            return "int32_t";
        }
        if (dataType == DataType.I64) {
            return "int64_t";
        }
        if (dataType == DataType.FP32) {
            return "float";
        }
        if (dataType == DataType.FP64) {
            return "double";
        }
        throw new UnsupportedOperationException("Unsupported HIP data type: " + dataType);
    }

    private String literal(Number value, DataType type) {
        if (type == DataType.BOOL) {
            int v = value.intValue() != 0 ? 1 : 0;
            return Integer.toString(v);
        }
        if (type == DataType.FP32) {
            return Float.toString(value.floatValue()) + "f";
        }
        if (type == DataType.FP64) {
            return Double.toString(value.doubleValue());
        }
        if (type == DataType.I64) {
            return value.longValue() + "LL";
        }
        if (type == DataType.I32 || type == DataType.I16 || type == DataType.I8) {
            return Integer.toString(value.intValue());
        }
        throw new UnsupportedOperationException("Unsupported literal type: " + type);
    }
}
