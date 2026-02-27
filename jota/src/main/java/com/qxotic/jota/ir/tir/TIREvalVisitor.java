package com.qxotic.jota.ir.tir;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.random.RandomAlgorithms;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

final class TIREvalVisitor implements TIRVisitor<MemoryView<MemorySegment>> {

    private final TIREvalContext context;
    private final MemoryAccess<MemorySegment> memAccess;

    TIREvalVisitor(TIREvalContext context) {
        this.context = context;
        this.memAccess = context.getMemoryAccess();
    }

    @Override
    public MemoryView<MemorySegment> visitTensorInput(TensorInput node) {
        return context.getInput(node.id());
    }

    @Override
    public MemoryView<MemorySegment> visitScalarInput(ScalarInput node) {
        return context.getInput(node.id());
    }

    @Override
    public MemoryView<MemorySegment> visitUnaryOp(UnaryOp node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);
        long size = layout.shape().size();

        if (dtype == DataType.FP16) {
            executeUnaryFP16(node.op(), input, output, size);
        } else if (dtype == DataType.BF16) {
            executeUnaryBF16(node.op(), input, output, size);
        } else if (dtype == DataType.FP32) {
            executeUnaryFloat(node.op(), input, output, size);
        } else if (dtype == DataType.FP64) {
            executeUnaryDouble(node.op(), input, output, size);
        } else if (dtype == DataType.I8) {
            executeUnaryByte(node.op(), input, output, size);
        } else if (dtype == DataType.I16) {
            executeUnaryShort(node.op(), input, output, size);
        } else if (dtype == DataType.I32) {
            executeUnaryInt(node.op(), input, output, size);
        } else if (dtype == DataType.I64) {
            executeUnaryLong(node.op(), input, output, size);
        } else if (dtype == DataType.BOOL) {
            executeUnaryBool(node.op(), input, output, size);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }

        return output;
    }

    private void executeUnaryFloat(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            float a = memAccess.readFloat(input.memory(), Indexing.linearToOffset(input, i));
            float result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> (float) Math.exp(a);
                        case LOG -> (float) Math.log(a);
                        case SQRT -> (float) Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> (float) Math.sin(a);
                        case COS -> (float) Math.cos(a);
                        case TAN -> (float) Math.tan(a);
                        case TANH -> (float) Math.tanh(a);
                        case RECIPROCAL -> 1.0f / a;
                        case LOGICAL_NOT ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_NOT not supported for FP32");
                        case BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_NOT not supported for FP32");
                    };
            memAccess.writeFloat(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryDouble(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            double a = memAccess.readDouble(input.memory(), Indexing.linearToOffset(input, i));
            double result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> Math.exp(a);
                        case LOG -> Math.log(a);
                        case SQRT -> Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> Math.sin(a);
                        case COS -> Math.cos(a);
                        case TAN -> Math.tan(a);
                        case TANH -> Math.tanh(a);
                        case RECIPROCAL -> 1.0 / a;
                        case LOGICAL_NOT ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_NOT not supported for FP64");
                        case BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_NOT not supported for FP64");
                    };
            memAccess.writeDouble(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryFP16(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short bits = memAccess.readShort(input.memory(), Indexing.linearToOffset(input, i));
            float a = Float.float16ToFloat(bits);
            float result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> (float) Math.exp(a);
                        case LOG -> (float) Math.log(a);
                        case SQRT -> (float) Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> (float) Math.sin(a);
                        case COS -> (float) Math.cos(a);
                        case TAN -> (float) Math.tan(a);
                        case TANH -> (float) Math.tanh(a);
                        case RECIPROCAL -> 1.0f / a;
                        case LOGICAL_NOT ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_NOT not supported for FP16");
                        case BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_NOT not supported for FP16");
                    };
            short resultBits = Float.floatToFloat16(result);
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), resultBits);
        }
    }

    private void executeUnaryBF16(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short bits = memAccess.readShort(input.memory(), Indexing.linearToOffset(input, i));
            float a = BFloat16.toFloat(bits);
            float result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case EXP -> (float) Math.exp(a);
                        case LOG -> (float) Math.log(a);
                        case SQRT -> (float) Math.sqrt(a);
                        case SQUARE -> a * a;
                        case SIN -> (float) Math.sin(a);
                        case COS -> (float) Math.cos(a);
                        case TAN -> (float) Math.tan(a);
                        case TANH -> (float) Math.tanh(a);
                        case RECIPROCAL -> 1.0f / a;
                        case LOGICAL_NOT ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_NOT not supported for BF16");
                        case BITWISE_NOT ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_NOT not supported for BF16");
                    };
            short resultBits = BFloat16.fromFloat(result);
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), resultBits);
        }
    }

    private void executeUnaryByte(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            byte a = memAccess.readByte(input.memory(), Indexing.linearToOffset(input, i));
            byte result =
                    switch (op) {
                        case NEGATE -> (byte) -a;
                        case ABS -> (byte) Math.abs(a);
                        case BITWISE_NOT -> (byte) ~a;
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported unary op for I8: " + op);
                    };
            memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryShort(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short a = memAccess.readShort(input.memory(), Indexing.linearToOffset(input, i));
            short result =
                    switch (op) {
                        case NEGATE -> (short) -a;
                        case ABS -> (short) Math.abs(a);
                        case BITWISE_NOT -> (short) ~a;
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported unary op for I16: " + op);
                    };
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryInt(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            int a = memAccess.readInt(input.memory(), Indexing.linearToOffset(input, i));
            int result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case BITWISE_NOT -> ~a;
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported unary op for I32: " + op);
                    };
            memAccess.writeInt(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryLong(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            long a = memAccess.readLong(input.memory(), Indexing.linearToOffset(input, i));
            long result =
                    switch (op) {
                        case NEGATE -> -a;
                        case ABS -> Math.abs(a);
                        case BITWISE_NOT -> ~a;
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported unary op for I64: " + op);
                    };
            memAccess.writeLong(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeUnaryBool(
            UnaryOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            byte a = memAccess.readByte(input.memory(), Indexing.linearToOffset(input, i));
            byte result =
                    switch (op) {
                        case LOGICAL_NOT -> (byte) (a != 0 ? 0 : 1);
                        default ->
                                throw new UnsupportedOperationException(
                                        "Unsupported unary op for BOOL: " + op);
                    };
            memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    @Override
    public MemoryView<MemorySegment> visitBinaryOp(BinaryOp node) {
        MemoryView<MemorySegment> left = context.evaluate(node.left());
        MemoryView<MemorySegment> right = context.evaluate(node.right());
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);
        long size = layout.shape().size();

        if (dtype == DataType.FP16) {
            executeBinaryFP16(node.op(), left, right, output, size);
        } else if (dtype == DataType.BF16) {
            executeBinaryBF16(node.op(), left, right, output, size);
        } else if (dtype == DataType.FP32) {
            executeBinaryFloat(node.op(), left, right, output, size);
        } else if (dtype == DataType.FP64) {
            executeBinaryDouble(node.op(), left, right, output, size);
        } else if (dtype == DataType.I8) {
            executeBinaryByte(node.op(), left, right, output, size);
        } else if (dtype == DataType.I16) {
            executeBinaryShort(node.op(), left, right, output, size);
        } else if (dtype == DataType.I32) {
            executeBinaryInt(node.op(), left, right, output, size);
        } else if (dtype == DataType.I64) {
            executeBinaryLong(node.op(), left, right, output, size);
        } else if (dtype == DataType.BOOL) {
            executeBinaryBool(node.op(), left, right, output, size);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }

        return output;
    }

    private int readShiftCount(MemoryView<MemorySegment> right, long linearIndex) {
        long offset = Indexing.linearToOffset(right, linearIndex);
        if (right.dataType() == DataType.I32) {
            return memAccess.readInt(right.memory(), offset);
        }
        if (right.dataType() == DataType.I8) {
            return memAccess.readByte(right.memory(), offset);
        }
        if (right.dataType() == DataType.I16) {
            return memAccess.readShort(right.memory(), offset);
        }
        if (right.dataType() == DataType.I64) {
            return (int) memAccess.readLong(right.memory(), offset);
        }
        throw new UnsupportedOperationException(
                "Shift count requires integral dtype, got " + right.dataType());
    }

    private void executeBinaryFloat(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            long leftOffset = Indexing.linearToOffset(left, i);
            long rightOffset = Indexing.linearToOffset(right, i);
            float a = memAccess.readFloat(left.memory(), leftOffset);
            float b = memAccess.readFloat(right.memory(), rightOffset);
            float result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> (float) Math.pow(a, b);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for FP32");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for FP32");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for FP32");
                        case BITWISE_AND ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_AND not supported for FP32");
                        case BITWISE_OR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_OR not supported for FP32");
                        case BITWISE_XOR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_XOR not supported for FP32");
                        case SHIFT_LEFT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_LEFT not supported for FP32");
                        case SHIFT_RIGHT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT not supported for FP32");
                        case SHIFT_RIGHT_UNSIGNED ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT_UNSIGNED not supported for FP32");
                        case EQUAL -> a == b ? 1.0f : 0.0f;
                        case LESS_THAN -> a < b ? 1.0f : 0.0f;
                    };
            memAccess.writeFloat(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryDouble(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            double a = memAccess.readDouble(left.memory(), Indexing.linearToOffset(left, i));
            double b = memAccess.readDouble(right.memory(), Indexing.linearToOffset(right, i));
            double result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> Math.pow(a, b);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for FP64");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for FP64");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for FP64");
                        case BITWISE_AND ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_AND not supported for FP64");
                        case BITWISE_OR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_OR not supported for FP64");
                        case BITWISE_XOR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_XOR not supported for FP64");
                        case SHIFT_LEFT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_LEFT not supported for FP64");
                        case SHIFT_RIGHT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT not supported for FP64");
                        case SHIFT_RIGHT_UNSIGNED ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT_UNSIGNED not supported for FP64");
                        case EQUAL -> a == b ? 1.0 : 0.0;
                        case LESS_THAN -> a < b ? 1.0 : 0.0;
                    };
            memAccess.writeDouble(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryFP16(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short aBits = memAccess.readShort(left.memory(), Indexing.linearToOffset(left, i));
            short bBits = memAccess.readShort(right.memory(), Indexing.linearToOffset(right, i));
            float a = Float.float16ToFloat(aBits);
            float b = Float.float16ToFloat(bBits);
            float result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> (float) Math.pow(a, b);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for FP16");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for FP16");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for FP16");
                        case BITWISE_AND ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_AND not supported for FP16");
                        case BITWISE_OR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_OR not supported for FP16");
                        case BITWISE_XOR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_XOR not supported for FP16");
                        case SHIFT_LEFT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_LEFT not supported for FP16");
                        case SHIFT_RIGHT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT not supported for FP16");
                        case SHIFT_RIGHT_UNSIGNED ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT_UNSIGNED not supported for FP16");
                        case EQUAL -> a == b ? 1.0f : 0.0f;
                        case LESS_THAN -> a < b ? 1.0f : 0.0f;
                    };
            short resultBits = Float.floatToFloat16(result);
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), resultBits);
        }
    }

    private void executeBinaryBF16(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short aBits = memAccess.readShort(left.memory(), Indexing.linearToOffset(left, i));
            short bBits = memAccess.readShort(right.memory(), Indexing.linearToOffset(right, i));
            float a = BFloat16.toFloat(aBits);
            float b = BFloat16.toFloat(bBits);
            float result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case POW -> (float) Math.pow(a, b);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for BF16");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for BF16");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for BF16");
                        case BITWISE_AND ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_AND not supported for BF16");
                        case BITWISE_OR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_OR not supported for BF16");
                        case BITWISE_XOR ->
                                throw new UnsupportedOperationException(
                                        "BITWISE_XOR not supported for BF16");
                        case SHIFT_LEFT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_LEFT not supported for BF16");
                        case SHIFT_RIGHT ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT not supported for BF16");
                        case SHIFT_RIGHT_UNSIGNED ->
                                throw new UnsupportedOperationException(
                                        "SHIFT_RIGHT_UNSIGNED not supported for BF16");
                        case EQUAL -> a == b ? 1.0f : 0.0f;
                        case LESS_THAN -> a < b ? 1.0f : 0.0f;
                    };
            short resultBits = BFloat16.fromFloat(result);
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), resultBits);
        }
    }

    private void executeBinaryByte(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            byte a = memAccess.readByte(left.memory(), Indexing.linearToOffset(left, i));
            byte b = memAccess.readByte(right.memory(), Indexing.linearToOffset(right, i));
            int shiftCount = readShiftCount(right, i);
            byte result =
                    switch (op) {
                        case ADD -> (byte) (a + b);
                        case SUBTRACT -> (byte) (a - b);
                        case MULTIPLY -> (byte) (a * b);
                        case DIVIDE -> (byte) (a / b);
                        case MIN -> (byte) Math.min(a, b);
                        case MAX -> (byte) Math.max(a, b);
                        case BITWISE_AND -> (byte) (a & b);
                        case BITWISE_OR -> (byte) (a | b);
                        case BITWISE_XOR -> (byte) (a ^ b);
                        case SHIFT_LEFT -> (byte) (a << (shiftCount & 7));
                        case SHIFT_RIGHT -> (byte) (a >> (shiftCount & 7));
                        case SHIFT_RIGHT_UNSIGNED -> (byte) ((a & 0xFF) >>> (shiftCount & 7));
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for I8");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for I8");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for I8");
                        case POW ->
                                throw new UnsupportedOperationException("POW not supported for I8");
                        case EQUAL -> (byte) (a == b ? 1 : 0);
                        case LESS_THAN -> (byte) (a < b ? 1 : 0);
                    };
            memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryShort(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            short a = memAccess.readShort(left.memory(), Indexing.linearToOffset(left, i));
            short b = memAccess.readShort(right.memory(), Indexing.linearToOffset(right, i));
            int shiftCount = readShiftCount(right, i);
            short result =
                    switch (op) {
                        case ADD -> (short) (a + b);
                        case SUBTRACT -> (short) (a - b);
                        case MULTIPLY -> (short) (a * b);
                        case DIVIDE -> (short) (a / b);
                        case MIN -> (short) Math.min(a, b);
                        case MAX -> (short) Math.max(a, b);
                        case BITWISE_AND -> (short) (a & b);
                        case BITWISE_OR -> (short) (a | b);
                        case BITWISE_XOR -> (short) (a ^ b);
                        case SHIFT_LEFT -> (short) (a << (shiftCount & 15));
                        case SHIFT_RIGHT -> (short) (a >> (shiftCount & 15));
                        case SHIFT_RIGHT_UNSIGNED -> (short) ((a & 0xFFFF) >>> (shiftCount & 15));
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for I16");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for I16");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for I16");
                        case POW ->
                                throw new UnsupportedOperationException(
                                        "POW not supported for I16");
                        case EQUAL -> (short) (a == b ? 1 : 0);
                        case LESS_THAN -> (short) (a < b ? 1 : 0);
                    };
            memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryInt(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            int a = memAccess.readInt(left.memory(), Indexing.linearToOffset(left, i));
            int b = memAccess.readInt(right.memory(), Indexing.linearToOffset(right, i));
            int shiftCount = readShiftCount(right, i);
            int result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case BITWISE_AND -> a & b;
                        case BITWISE_OR -> a | b;
                        case BITWISE_XOR -> a ^ b;
                        case SHIFT_LEFT -> a << (shiftCount & 31);
                        case SHIFT_RIGHT -> a >> (shiftCount & 31);
                        case SHIFT_RIGHT_UNSIGNED -> a >>> (shiftCount & 31);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for I32");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for I32");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for I32");
                        case POW ->
                                throw new UnsupportedOperationException(
                                        "POW not supported for I32");
                        case EQUAL -> a == b ? 1 : 0;
                        case LESS_THAN -> a < b ? 1 : 0;
                    };
            memAccess.writeInt(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryLong(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            long a = memAccess.readLong(left.memory(), Indexing.linearToOffset(left, i));
            long b = memAccess.readLong(right.memory(), Indexing.linearToOffset(right, i));
            int shiftCount = readShiftCount(right, i);
            long result =
                    switch (op) {
                        case ADD -> a + b;
                        case SUBTRACT -> a - b;
                        case MULTIPLY -> a * b;
                        case DIVIDE -> a / b;
                        case MIN -> Math.min(a, b);
                        case MAX -> Math.max(a, b);
                        case BITWISE_AND -> a & b;
                        case BITWISE_OR -> a | b;
                        case BITWISE_XOR -> a ^ b;
                        case SHIFT_LEFT -> a << (shiftCount & 63);
                        case SHIFT_RIGHT -> a >> (shiftCount & 63);
                        case SHIFT_RIGHT_UNSIGNED -> a >>> (shiftCount & 63);
                        case LOGICAL_AND ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_AND not supported for I64");
                        case LOGICAL_OR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_OR not supported for I64");
                        case LOGICAL_XOR ->
                                throw new UnsupportedOperationException(
                                        "LOGICAL_XOR not supported for I64");
                        case POW ->
                                throw new UnsupportedOperationException(
                                        "POW not supported for I64");
                        case EQUAL -> a == b ? 1L : 0L;
                        case LESS_THAN -> a < b ? 1L : 0L;
                    };
            memAccess.writeLong(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    private void executeBinaryBool(
            BinaryOperator op,
            MemoryView<MemorySegment> left,
            MemoryView<MemorySegment> right,
            MemoryView<MemorySegment> output,
            long size) {
        for (long i = 0; i < size; i++) {
            byte a = memAccess.readByte(left.memory(), Indexing.linearToOffset(left, i));
            byte b = memAccess.readByte(right.memory(), Indexing.linearToOffset(right, i));
            byte result =
                    switch (op) {
                        case LOGICAL_AND -> (byte) ((a != 0 && b != 0) ? 1 : 0);
                        case LOGICAL_OR -> (byte) ((a != 0 || b != 0) ? 1 : 0);
                        case LOGICAL_XOR -> (byte) ((a != 0) != (b != 0) ? 1 : 0);
                        case ADD, SUBTRACT, MULTIPLY, DIVIDE, MIN, MAX, POW ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for BOOL");
                        case BITWISE_AND,
                                BITWISE_OR,
                                BITWISE_XOR,
                                SHIFT_LEFT,
                                SHIFT_RIGHT,
                                SHIFT_RIGHT_UNSIGNED ->
                                throw new UnsupportedOperationException(
                                        op + " not supported for BOOL");
                        case EQUAL -> (byte) (((a != 0) == (b != 0)) ? 1 : 0);
                        case LESS_THAN -> (byte) (((a != 0 ? 1 : 0) < (b != 0 ? 1 : 0)) ? 1 : 0);
                    };
            memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), result);
        }
    }

    @Override
    public MemoryView<MemorySegment> visitTernaryOp(TernaryOp node) {
        MemoryView<MemorySegment> cond = context.evaluate(node.cond());
        MemoryView<MemorySegment> trueExpr = context.evaluate(node.trueExpr());
        MemoryView<MemorySegment> falseExpr = context.evaluate(node.falseExpr());
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);
        long size = layout.shape().size();

        for (long i = 0; i < size; i++) {
            boolean condition =
                    memAccess.readByte(cond.memory(), Indexing.linearToOffset(cond, i)) != 0;

            if (dtype == DataType.FP32) {
                float val =
                        condition
                                ? memAccess.readFloat(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readFloat(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeFloat(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.FP64) {
                double val =
                        condition
                                ? memAccess.readDouble(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readDouble(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeDouble(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
                short val =
                        condition
                                ? memAccess.readShort(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readShort(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.I8) {
                byte val =
                        condition
                                ? memAccess.readByte(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readByte(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.I16) {
                short val =
                        condition
                                ? memAccess.readShort(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readShort(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.I32) {
                int val =
                        condition
                                ? memAccess.readInt(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readInt(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeInt(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.I64) {
                long val =
                        condition
                                ? memAccess.readLong(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readLong(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeLong(output.memory(), Indexing.linearToOffset(output, i), val);
            } else if (dtype == DataType.BOOL) {
                byte val =
                        condition
                                ? memAccess.readByte(
                                        trueExpr.memory(), Indexing.linearToOffset(trueExpr, i))
                                : memAccess.readByte(
                                        falseExpr.memory(), Indexing.linearToOffset(falseExpr, i));
                memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), val);
            } else {
                throw new UnsupportedOperationException("Unsupported data type: " + dtype);
            }
        }

        return output;
    }

    @Override
    public MemoryView<MemorySegment> visitCastOp(CastOp node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());
        Layout layout = Layout.rowMajor(node.shape());
        DataType sourceDtype = node.input().dataType();
        DataType targetDtype = node.targetDataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(targetDtype, layout);
        long size = layout.shape().size();

        for (long i = 0; i < size; i++) {
            long inOffset = Indexing.linearToOffset(input, i);
            long outOffset = Indexing.linearToOffset(output, i);

            if (targetDtype == DataType.FP32) {
                float value = readAsFloat(input, sourceDtype, inOffset);
                memAccess.writeFloat(output.memory(), outOffset, value);
            } else if (targetDtype == DataType.FP64) {
                double value = readAsDouble(input, sourceDtype, inOffset);
                memAccess.writeDouble(output.memory(), outOffset, value);
            } else if (targetDtype == DataType.FP16) {
                float value = readAsFloat(input, sourceDtype, inOffset);
                short bits = Float.floatToFloat16(value);
                memAccess.writeShort(output.memory(), outOffset, bits);
            } else if (targetDtype == DataType.BF16) {
                float value = readAsFloat(input, sourceDtype, inOffset);
                short bits = BFloat16.fromFloat(value);
                memAccess.writeShort(output.memory(), outOffset, bits);
            } else if (targetDtype == DataType.I8) {
                long value = readAsLong(input, sourceDtype, inOffset);
                memAccess.writeByte(output.memory(), outOffset, (byte) value);
            } else if (targetDtype == DataType.I16) {
                long value = readAsLong(input, sourceDtype, inOffset);
                memAccess.writeShort(output.memory(), outOffset, (short) value);
            } else if (targetDtype == DataType.I32) {
                long value = readAsLong(input, sourceDtype, inOffset);
                memAccess.writeInt(output.memory(), outOffset, (int) value);
            } else if (targetDtype == DataType.I64) {
                long value = readAsLong(input, sourceDtype, inOffset);
                memAccess.writeLong(output.memory(), outOffset, value);
            } else if (targetDtype == DataType.BOOL) {
                long value = readAsLong(input, sourceDtype, inOffset);
                memAccess.writeByte(output.memory(), outOffset, (byte) (value != 0 ? 1 : 0));
            } else {
                throw new UnsupportedOperationException(
                        "Unsupported target data type: " + targetDtype);
            }
        }

        return output;
    }

    private float readAsFloat(MemoryView<MemorySegment> view, DataType dtype, long offset) {
        if (dtype == DataType.FP32) {
            return memAccess.readFloat(view.memory(), offset);
        } else if (dtype == DataType.FP64) {
            return (float) memAccess.readDouble(view.memory(), offset);
        } else if (dtype == DataType.FP16) {
            return Float.float16ToFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.BF16) {
            return BFloat16.toFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.I8) {
            return memAccess.readByte(view.memory(), offset);
        } else if (dtype == DataType.I16) {
            return memAccess.readShort(view.memory(), offset);
        } else if (dtype == DataType.I32) {
            return memAccess.readInt(view.memory(), offset);
        } else if (dtype == DataType.I64) {
            return memAccess.readLong(view.memory(), offset);
        } else if (dtype == DataType.BOOL) {
            return memAccess.readByte(view.memory(), offset) != 0 ? 1.0f : 0.0f;
        } else {
            throw new UnsupportedOperationException("Cannot convert to float from: " + dtype);
        }
    }

    private double readAsDouble(MemoryView<MemorySegment> view, DataType dtype, long offset) {
        if (dtype == DataType.FP32) {
            return memAccess.readFloat(view.memory(), offset);
        } else if (dtype == DataType.FP64) {
            return memAccess.readDouble(view.memory(), offset);
        } else if (dtype == DataType.FP16) {
            return Float.float16ToFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.BF16) {
            return BFloat16.toFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.I8) {
            return memAccess.readByte(view.memory(), offset);
        } else if (dtype == DataType.I16) {
            return memAccess.readShort(view.memory(), offset);
        } else if (dtype == DataType.I32) {
            return memAccess.readInt(view.memory(), offset);
        } else if (dtype == DataType.I64) {
            return memAccess.readLong(view.memory(), offset);
        } else if (dtype == DataType.BOOL) {
            return memAccess.readByte(view.memory(), offset) != 0 ? 1.0 : 0.0;
        } else {
            throw new UnsupportedOperationException("Cannot convert to double from: " + dtype);
        }
    }

    private long readAsLong(MemoryView<MemorySegment> view, DataType dtype, long offset) {
        if (dtype == DataType.FP32) {
            return (long) memAccess.readFloat(view.memory(), offset);
        } else if (dtype == DataType.FP64) {
            return (long) memAccess.readDouble(view.memory(), offset);
        } else if (dtype == DataType.FP16) {
            return (long) Float.float16ToFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.BF16) {
            return (long) BFloat16.toFloat(memAccess.readShort(view.memory(), offset));
        } else if (dtype == DataType.I8) {
            return memAccess.readByte(view.memory(), offset);
        } else if (dtype == DataType.I16) {
            return memAccess.readShort(view.memory(), offset);
        } else if (dtype == DataType.I32) {
            return memAccess.readInt(view.memory(), offset);
        } else if (dtype == DataType.I64) {
            return memAccess.readLong(view.memory(), offset);
        } else if (dtype == DataType.BOOL) {
            return memAccess.readByte(view.memory(), offset);
        } else {
            throw new UnsupportedOperationException("Cannot convert to long from: " + dtype);
        }
    }

    @Override
    public MemoryView<MemorySegment> visitReductionOp(ReductionOp node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());
        Layout inputLayout = input.layout();
        Layout outputLayout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, outputLayout);

        if (dtype == DataType.FP32) {
            executeReductionFloat(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.FP64) {
            executeReductionDouble(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.FP16) {
            executeReductionFP16(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.BF16) {
            executeReductionBF16(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.I8) {
            executeReductionByte(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.I16) {
            executeReductionShort(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.I32) {
            executeReductionInt(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.I64) {
            executeReductionLong(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else if (dtype == DataType.BOOL) {
            executeReductionBool(
                    node.op(),
                    input,
                    output,
                    inputLayout.shape(),
                    outputLayout.shape(),
                    node.axes());
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }

        return output;
    }

    private void executeReductionFloat(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            float acc =
                    switch (op) {
                        case SUM -> 0.0f;
                        case PROD -> 1.0f;
                        case MIN -> Float.MAX_VALUE;
                        case MAX -> -Float.MAX_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                float value = memAccess.readFloat(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset = Indexing.linearToOffset(output, outIdx);
            memAccess.writeFloat(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionDouble(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            double acc =
                    switch (op) {
                        case SUM -> 0.0;
                        case PROD -> 1.0;
                        case MIN -> Double.POSITIVE_INFINITY;
                        case MAX -> Double.NEGATIVE_INFINITY;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                double value = memAccess.readDouble(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset = Indexing.linearToOffset(output, outIdx);
            memAccess.writeDouble(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionFP16(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            float acc =
                    switch (op) {
                        case SUM -> 0.0f;
                        case PROD -> 1.0f;
                        case MIN -> Float.MAX_VALUE;
                        case MAX -> -Float.MAX_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                short bits = memAccess.readShort(input.memory(), inOffset);
                float value = Float.float16ToFloat(bits);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.FP16, outIdx);
            short resultBits = Float.floatToFloat16(acc);
            memAccess.writeShort(output.memory(), outOffset, resultBits);
        }
    }

    private void executeReductionBF16(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            float acc =
                    switch (op) {
                        case SUM -> 0.0f;
                        case PROD -> 1.0f;
                        case MIN -> Float.MAX_VALUE;
                        case MAX -> -Float.MAX_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                short bits = memAccess.readShort(input.memory(), inOffset);
                float value = BFloat16.toFloat(bits);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.BF16, outIdx);
            short resultBits = BFloat16.fromFloat(acc);
            memAccess.writeShort(output.memory(), outOffset, resultBits);
        }
    }

    private void executeReductionByte(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            byte acc =
                    switch (op) {
                        case SUM -> (byte) 0;
                        case PROD -> (byte) 1;
                        case MIN -> Byte.MAX_VALUE;
                        case MAX -> Byte.MIN_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                byte value = memAccess.readByte(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> (byte) (acc + value);
                            case PROD -> (byte) (acc * value);
                            case MIN -> (byte) Math.min(acc, value);
                            case MAX -> (byte) Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.I8, outIdx);
            memAccess.writeByte(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionShort(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            short acc =
                    switch (op) {
                        case SUM -> (short) 0;
                        case PROD -> (short) 1;
                        case MIN -> Short.MAX_VALUE;
                        case MAX -> Short.MIN_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                short value = memAccess.readShort(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> (short) (acc + value);
                            case PROD -> (short) (acc * value);
                            case MIN -> (short) Math.min(acc, value);
                            case MAX -> (short) Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.I16, outIdx);
            memAccess.writeShort(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionInt(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            int acc =
                    switch (op) {
                        case SUM -> 0;
                        case PROD -> 1;
                        case MIN -> Integer.MAX_VALUE;
                        case MAX -> Integer.MIN_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                int value = memAccess.readInt(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.I32, outIdx);
            memAccess.writeInt(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionLong(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            long acc =
                    switch (op) {
                        case SUM -> 0L;
                        case PROD -> 1L;
                        case MIN -> Long.MAX_VALUE;
                        case MAX -> Long.MIN_VALUE;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                long value = memAccess.readLong(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> acc + value;
                            case PROD -> acc * value;
                            case MIN -> Math.min(acc, value);
                            case MAX -> Math.max(acc, value);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.I64, outIdx);
            memAccess.writeLong(output.memory(), outOffset, acc);
        }
    }

    private void executeReductionBool(
            ReductionOperator op,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> output,
            Shape inputShape,
            Shape outputShape,
            int[] axes) {
        long outputSize = outputShape.size();

        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            byte acc =
                    switch (op) {
                        case SUM -> (byte) 0;
                        case PROD -> (byte) 1;
                        case MIN -> (byte) 1;
                        case MAX -> (byte) 0;
                    };

            for (long[] inCoord :
                    iterateReduction(
                            inputShape, axes, Indexing.linearToCoord(outputShape, outIdx))) {
                long inOffset = Indexing.coordToOffset(input, inCoord);
                byte value = memAccess.readByte(input.memory(), inOffset);
                acc =
                        switch (op) {
                            case SUM -> (byte) (((acc != 0) ? 1 : 0) + ((value != 0) ? 1 : 0));
                            case PROD -> (byte) (((acc != 0) && (value != 0)) ? 1 : 0);
                            case MIN -> (byte) (((acc != 0) && (value != 0)) ? 1 : 0);
                            case MAX -> (byte) (((acc != 0) || (value != 0)) ? 1 : 0);
                        };
            }

            long outOffset =
                    Indexing.linearToOffset(outputShape, output.stride(), DataType.BOOL, outIdx);
            memAccess.writeByte(output.memory(), outOffset, acc);
        }
    }

    private Iterable<long[]> iterateReduction(Shape inputShape, int[] axes, long[] outCoord) {
        List<long[]> result = new ArrayList<>();
        generateReductionCoords(
                result, new long[inputShape.flatRank()], inputShape, axes, outCoord, 0);
        return result;
    }

    private void generateReductionCoords(
            List<long[]> result,
            long[] inCoord,
            Shape inputShape,
            int[] axes,
            long[] outCoord,
            int axisIdx) {
        if (axisIdx == inputShape.flatRank()) {
            result.add(inCoord.clone());
            return;
        }

        if (isReduced(axisIdx, axes)) {
            long dim = inputShape.flatAt(axisIdx);
            for (long i = 0; i < dim; i++) {
                inCoord[axisIdx] = i;
                generateReductionCoords(result, inCoord, inputShape, axes, outCoord, axisIdx + 1);
            }
        } else {
            int outIdx = getOutIdx(axisIdx, axes);
            inCoord[axisIdx] = outCoord[outIdx];
            generateReductionCoords(result, inCoord, inputShape, axes, outCoord, axisIdx + 1);
        }
    }

    private boolean isReduced(int axis, int[] axes) {
        for (int a : axes) {
            if (a == axis) {
                return true;
            }
        }
        return false;
    }

    private int getOutIdx(int axis, int[] axes) {
        int count = 0;
        for (int a : axes) {
            if (a < axis) {
                count++;
            }
        }
        return axis - count;
    }

    @Override
    public MemoryView<MemorySegment> visitViewTransform(ViewTransform node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());

        // If lazy indexing is needed (e.g., transpose + reshape), we must copy to contiguous first
        // because the strides in node.layout() are placeholders.
        if (node.needsLazyIndexing()) {
            // Force contiguous copy, then apply the view with correct layout
            MemoryView<MemorySegment> contiguous = copyToContiguous(input);
            // After copy, data is row-major, so we can use row-major strides for the new shape
            return MemoryView.of(
                    contiguous.memory(),
                    contiguous.byteOffset(),
                    node.dataType(),
                    Layout.rowMajor(node.layout().shape()));
        }

        return MemoryView.of(input.memory(), input.byteOffset(), node.dataType(), node.layout());
    }

    private MemoryView<MemorySegment> copyToContiguous(MemoryView<MemorySegment> input) {
        // Match eager contiguous() semantics: require row-major suffix contiguity,
        // not just spanning a contiguous byte range (broadcast/transpose views may fail linear
        // access).
        if (input.layout().isSuffixContiguous(0)) {
            return input;
        }
        Layout rowMajor = Layout.rowMajor(input.shape());
        MemoryView<MemorySegment> output = context.allocateTemporary(input.dataType(), rowMajor);
        long size = input.shape().size();
        DataType dtype = input.dataType();

        for (long i = 0; i < size; i++) {
            long inOffset = Indexing.linearToOffset(input, i);
            long outOffset = i * dtype.byteSize();
            copyElement(input, inOffset, output, outOffset, dtype);
        }
        return output;
    }

    private void copyElement(
            MemoryView<MemorySegment> src,
            long srcOffset,
            MemoryView<MemorySegment> dst,
            long dstOffset,
            DataType dtype) {
        if (dtype == DataType.FP32) {
            float val = memAccess.readFloat(src.memory(), srcOffset);
            memAccess.writeFloat(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.FP64) {
            double val = memAccess.readDouble(src.memory(), srcOffset);
            memAccess.writeDouble(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.I32) {
            int val = memAccess.readInt(src.memory(), srcOffset);
            memAccess.writeInt(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.I64) {
            long val = memAccess.readLong(src.memory(), srcOffset);
            memAccess.writeLong(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.I8) {
            byte val = memAccess.readByte(src.memory(), srcOffset);
            memAccess.writeByte(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.I16) {
            short val = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.FP16) {
            short val = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.BF16) {
            short val = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, val);
        } else if (dtype == DataType.BOOL) {
            byte val = memAccess.readByte(src.memory(), srcOffset);
            memAccess.writeByte(dst.memory(), dstOffset, val);
        } else {
            throw new UnsupportedOperationException("Unsupported dtype: " + dtype);
        }
    }

    @Override
    public MemoryView<MemorySegment> visitContiguous(Contiguous node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());

        if (input.isContiguous()) {
            return input;
        }

        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);

        long size = layout.shape().size();
        for (long i = 0; i < size; i++) {
            long inOffset = Indexing.linearToOffset(input, i);
            long outOffset = Indexing.linearToOffset(output, i);
            copyElement(input.memory(), inOffset, output.memory(), outOffset, dtype);
        }

        return output;
    }

    private void copyElement(
            Memory<MemorySegment> src,
            long srcOffset,
            Memory<MemorySegment> dst,
            long dstOffset,
            DataType dtype) {
        if (dtype == DataType.FP32) {
            memAccess.writeFloat(dst, dstOffset, memAccess.readFloat(src, srcOffset));
        } else if (dtype == DataType.FP64) {
            memAccess.writeDouble(dst, dstOffset, memAccess.readDouble(src, srcOffset));
        } else if (dtype == DataType.FP16 || dtype == DataType.BF16 || dtype == DataType.I16) {
            memAccess.writeShort(dst, dstOffset, memAccess.readShort(src, srcOffset));
        } else if (dtype == DataType.I32) {
            memAccess.writeInt(dst, dstOffset, memAccess.readInt(src, srcOffset));
        } else if (dtype == DataType.I64) {
            memAccess.writeLong(dst, dstOffset, memAccess.readLong(src, srcOffset));
        } else if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            memAccess.writeByte(dst, dstOffset, memAccess.readByte(src, srcOffset));
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }
    }

    @Override
    public MemoryView<MemorySegment> visitScalarConstant(ScalarConstant node) {
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);
        long size = layout.shape().size();
        for (long i = 0; i < size; i++) {
            long offset = Indexing.linearToOffset(output, i);
            if (dtype == DataType.FP32) {
                memAccess.writeFloat(
                        output.memory(), offset, Float.intBitsToFloat((int) node.rawBits()));
            } else if (dtype == DataType.FP64) {
                memAccess.writeDouble(
                        output.memory(), offset, Double.longBitsToDouble(node.rawBits()));
            } else if (dtype == DataType.FP16 || dtype == DataType.BF16) {
                memAccess.writeShort(output.memory(), offset, (short) node.rawBits());
            } else if (dtype == DataType.I8) {
                memAccess.writeByte(output.memory(), offset, (byte) node.rawBits());
            } else if (dtype == DataType.I16) {
                memAccess.writeShort(output.memory(), offset, (short) node.rawBits());
            } else if (dtype == DataType.I32) {
                memAccess.writeInt(output.memory(), offset, (int) node.rawBits());
            } else if (dtype == DataType.I64) {
                memAccess.writeLong(output.memory(), offset, node.rawBits());
            } else if (dtype == DataType.BOOL) {
                memAccess.writeByte(output.memory(), offset, (byte) (node.rawBits() != 0 ? 1 : 0));
            } else {
                throw new UnsupportedOperationException("Unsupported data type: " + dtype);
            }
        }
        return output;
    }

    @Override
    public MemoryView<MemorySegment> visitIotaConstant(IotaConstant node) {
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        long count = node.count();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);

        for (long i = 0; i < count; i++) {
            if (dtype == DataType.FP32) {
                memAccess.writeFloat(
                        output.memory(), Indexing.linearToOffset(output, i), (float) i);
            } else if (dtype == DataType.FP64) {
                memAccess.writeDouble(
                        output.memory(), Indexing.linearToOffset(output, i), (double) i);
            } else if (dtype == DataType.FP16) {
                short bits = Float.floatToFloat16((float) i);
                memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), bits);
            } else if (dtype == DataType.BF16) {
                short bits = BFloat16.fromFloat((float) i);
                memAccess.writeShort(output.memory(), Indexing.linearToOffset(output, i), bits);
            } else if (dtype == DataType.I8) {
                memAccess.writeByte(output.memory(), Indexing.linearToOffset(output, i), (byte) i);
            } else if (dtype == DataType.I16) {
                memAccess.writeShort(
                        output.memory(), Indexing.linearToOffset(output, i), (short) i);
            } else if (dtype == DataType.I32) {
                memAccess.writeInt(output.memory(), Indexing.linearToOffset(output, i), (int) i);
            } else if (dtype == DataType.I64) {
                memAccess.writeLong(output.memory(), Indexing.linearToOffset(output, i), i);
            } else if (dtype == DataType.BOOL) {
                memAccess.writeByte(
                        output.memory(),
                        Indexing.linearToOffset(output, i),
                        (byte) (i != 0 ? 1 : 0));
            } else {
                throw new UnsupportedOperationException("Unsupported data type: " + dtype);
            }
        }

        return output;
    }

    @Override
    public MemoryView<MemorySegment> visitRandomUniformOp(RandomUniformOp node) {
        Layout layout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, layout);
        long count = node.shape().size();

        for (long i = 0; i < count; i++) {
            if (dtype == DataType.FP32) {
                float value = RandomAlgorithms.uniformFp32(i, node.key0(), node.key1());
                memAccess.writeFloat(output.memory(), Indexing.linearToOffset(output, i), value);
            } else if (dtype == DataType.FP64) {
                double value = RandomAlgorithms.uniformFp64(i, node.key0(), node.key1());
                memAccess.writeDouble(output.memory(), Indexing.linearToOffset(output, i), value);
            } else {
                throw new UnsupportedOperationException(
                        "RandomUniformOp supports FP32/FP64 only, got: " + dtype);
            }
        }
        return output;
    }

    @Override
    public MemoryView<MemorySegment> visitGatherOp(GatherOp node) {
        MemoryView<MemorySegment> input = context.evaluate(node.input());
        MemoryView<MemorySegment> indices = context.evaluate(node.indices());

        Layout outputLayout = Layout.rowMajor(node.shape());
        DataType dtype = node.dataType();
        DataType indicesType = indices.dataType();
        MemoryView<MemorySegment> output = context.allocateTemporary(dtype, outputLayout);

        int axis = node.axis();
        int inputRank = input.shape().rank();
        int normalizedAxis = Util.wrapAround(axis, inputRank);

        long outputSize = node.shape().size();

        // Iterate over all output positions
        for (long outIdx = 0; outIdx < outputSize; outIdx++) {
            long[] outCoord = Indexing.linearToCoord(outputLayout.shape(), outIdx);

            // Build input coordinate from output coordinate
            // Output shape: [input[0:axis], indices..., input[axis+1:]]
            // For embedding lookup: input=[V, H], indices=[B, S], axis=0, output=[B, S, H]
            // outCoord = [b, s, h], gather axis=0
            // inputCoord = [indices[b,s], h]

            long[] inCoord = new long[inputRank];
            int indicesRank = indices.shape().rank();

            // Fill dimensions before axis: copy directly from output
            for (int i = 0; i < normalizedAxis; i++) {
                inCoord[i] = outCoord[i];
            }

            // At axis position: lookup in indices tensor
            // Indices dimensions in output start at position 'normalizedAxis'
            long[] idxCoord = new long[indicesRank];
            for (int j = 0; j < indicesRank; j++) {
                idxCoord[j] = outCoord[normalizedAxis + j];
            }
            long idxOffset = Indexing.coordToOffset(indices, idxCoord);
            inCoord[normalizedAxis] = readIndexValue(indices, idxOffset, indicesType);

            // Fill dimensions after axis: copy from output after indices dimensions
            for (int i = normalizedAxis + 1; i < inputRank; i++) {
                inCoord[i] = outCoord[normalizedAxis + indicesRank + (i - normalizedAxis - 1)];
            }

            // Read from input and write to output
            long inOffset = Indexing.coordToOffset(input, inCoord);
            long outOffset = Indexing.linearToOffset(output, outIdx);

            copyValue(input, inOffset, output, outOffset, dtype);
        }

        return output;
    }

    private long readIndexValue(MemoryView<MemorySegment> indices, long offset, DataType dtype) {
        if (dtype == DataType.I8) {
            return memAccess.readByte(indices.memory(), offset);
        } else if (dtype == DataType.I16) {
            return memAccess.readShort(indices.memory(), offset);
        } else if (dtype == DataType.I32) {
            return memAccess.readInt(indices.memory(), offset);
        } else if (dtype == DataType.I64) {
            return memAccess.readLong(indices.memory(), offset);
        } else {
            throw new UnsupportedOperationException("Unsupported index data type: " + dtype);
        }
    }

    private void copyValue(
            MemoryView<MemorySegment> src,
            long srcOffset,
            MemoryView<MemorySegment> dst,
            long dstOffset,
            DataType dtype) {
        if (dtype == DataType.I8 || dtype == DataType.BOOL) {
            byte v = memAccess.readByte(src.memory(), srcOffset);
            memAccess.writeByte(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.I16) {
            short v = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.I32) {
            int v = memAccess.readInt(src.memory(), srcOffset);
            memAccess.writeInt(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.I64) {
            long v = memAccess.readLong(src.memory(), srcOffset);
            memAccess.writeLong(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.FP32) {
            float v = memAccess.readFloat(src.memory(), srcOffset);
            memAccess.writeFloat(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.FP64) {
            double v = memAccess.readDouble(src.memory(), srcOffset);
            memAccess.writeDouble(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.FP16) {
            short v = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, v);
        } else if (dtype == DataType.BF16) {
            short v = memAccess.readShort(src.memory(), srcOffset);
            memAccess.writeShort(dst.memory(), dstOffset, v);
        } else {
            throw new UnsupportedOperationException("Unsupported data type: " + dtype);
        }
    }
}
