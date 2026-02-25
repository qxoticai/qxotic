package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.TypeRules;
import com.qxotic.jota.memory.MemoryView;
import java.util.List;
import org.junit.jupiter.api.Test;

class TensorOperationTypeMatrixTest {

    private static final List<DataType> PRIMITIVE_TYPES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    @FunctionalInterface
    private interface BinaryTensorOp {
        Tensor apply(Tensor left, Tensor right);
    }

    @Test
    void arithmeticOpsRespectTypePromotionMatrix() {
        List<BinaryTensorOp> ops =
                List.of(
                        Tensor::add,
                        Tensor::subtract,
                        Tensor::multiply,
                        Tensor::divide,
                        Tensor::min,
                        Tensor::max);

        for (BinaryTensorOp op : ops) {
            for (DataType leftType : PRIMITIVE_TYPES) {
                for (DataType rightType : PRIMITIVE_TYPES) {
                    Tensor left = scalar(leftType, 3);
                    Tensor right = scalar(rightType, 2);
                    DataType expected;
                    try {
                        expected = TypeRules.promote(leftType, rightType);
                    } catch (IllegalArgumentException e) {
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> op.apply(left, right).materialize(),
                                () -> "Expected rejection for " + leftType + " and " + rightType);
                        continue;
                    }
                    MemoryView<?> out = op.apply(left, right).materialize();
                    assertEquals(expected, out.dataType());
                }
            }
        }
    }

    @Test
    void comparisonOpsRespectTypePromotionMatrix() {
        List<BinaryTensorOp> ops =
                List.of(
                        Tensor::equal,
                        Tensor::lessThan,
                        Tensor::notEqual,
                        Tensor::greaterThan,
                        Tensor::lessThanOrEqual,
                        Tensor::greaterThanOrEqual);

        for (BinaryTensorOp op : ops) {
            for (DataType leftType : PRIMITIVE_TYPES) {
                for (DataType rightType : PRIMITIVE_TYPES) {
                    Tensor left = scalar(leftType, 3);
                    Tensor right = scalar(rightType, 2);
                    try {
                        TypeRules.promoteForComparison(leftType, rightType);
                    } catch (IllegalArgumentException e) {
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> op.apply(left, right).materialize(),
                                () -> "Expected rejection for " + leftType + " and " + rightType);
                        continue;
                    }
                    MemoryView<?> out = op.apply(left, right).materialize();
                    assertEquals(DataType.BOOL, out.dataType());
                }
            }
        }
    }

    @Test
    void logicalOpsRequireBoolPairs() {
        List<BinaryTensorOp> ops =
                List.of(Tensor::logicalAnd, Tensor::logicalOr, Tensor::logicalXor);
        for (BinaryTensorOp op : ops) {
            for (DataType leftType : PRIMITIVE_TYPES) {
                for (DataType rightType : PRIMITIVE_TYPES) {
                    Tensor left = scalar(leftType, 1);
                    Tensor right = scalar(rightType, 0);
                    if (leftType == DataType.BOOL && rightType == DataType.BOOL) {
                        MemoryView<?> out = op.apply(left, right).materialize();
                        assertEquals(DataType.BOOL, out.dataType());
                    } else {
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> op.apply(left, right).materialize(),
                                () ->
                                        "Expected BOOL-only rejection for "
                                                + leftType
                                                + " and "
                                                + rightType);
                    }
                }
            }
        }
    }

    @Test
    void bitwiseOpsRequireSameIntegralNonBoolTypes() {
        List<BinaryTensorOp> ops =
                List.of(Tensor::bitwiseAnd, Tensor::bitwiseOr, Tensor::bitwiseXor);
        for (BinaryTensorOp op : ops) {
            for (DataType leftType : PRIMITIVE_TYPES) {
                for (DataType rightType : PRIMITIVE_TYPES) {
                    Tensor left = scalar(leftType, 1);
                    Tensor right = scalar(rightType, 3);
                    boolean supported =
                            leftType == rightType
                                    && leftType.isIntegral()
                                    && leftType != DataType.BOOL;
                    if (supported) {
                        MemoryView<?> out = op.apply(left, right).materialize();
                        assertEquals(leftType, out.dataType());
                    } else {
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> op.apply(left, right).materialize(),
                                () ->
                                        "Expected bitwise rejection for "
                                                + leftType
                                                + " and "
                                                + rightType);
                    }
                }
            }
        }
    }

    @Test
    void bitwiseNotRequiresIntegralNonBoolType() {
        for (DataType type : PRIMITIVE_TYPES) {
            Tensor input = scalar(type, 1);
            if (type.isIntegral() && type != DataType.BOOL) {
                MemoryView<?> out = input.bitwiseNot().materialize();
                assertEquals(type, out.dataType());
            } else {
                assertThrows(
                        IllegalArgumentException.class, () -> input.bitwiseNot().materialize());
            }
        }
    }

    @Test
    void whereTypeRulesAreEnforced() {
        Tensor condition = scalar(DataType.BOOL, 1);
        for (DataType type : PRIMITIVE_TYPES) {
            Tensor value = scalar(type, 7);
            MemoryView<?> out = condition.where(value, value).materialize();
            assertEquals(type, out.dataType());
        }

        Tensor nonBoolCondition = scalar(DataType.I32, 1);
        assertThrows(
                IllegalArgumentException.class,
                () -> nonBoolCondition.where(scalar(DataType.I32, 1), scalar(DataType.I32, 2)));
        assertThrows(
                IllegalArgumentException.class,
                () -> condition.where(scalar(DataType.I32, 1), scalar(DataType.FP32, 2)));
    }

    @Test
    void anyAllRequireBoolInputs() {
        for (DataType type : PRIMITIVE_TYPES) {
            Tensor input = scalar(type, 1);
            if (type == DataType.BOOL) {
                assertEquals(DataType.BOOL, input.any().materialize().dataType());
                assertEquals(DataType.BOOL, input.all().materialize().dataType());
            } else {
                assertThrows(IllegalArgumentException.class, input::any);
                assertThrows(IllegalArgumentException.class, input::all);
            }
        }
    }

    private static Tensor scalar(DataType type, long value) {
        return Tensor.scalar(value, type);
    }
}
