package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

/**
 * TIR pass that folds constant expressions at compile time.
 *
 * <p>This pass handles:
 *
 * <ul>
 *   <li>Binary operations on two ScalarConstants → ScalarConstant
 *   <li>Unary operations on ScalarConstant → ScalarConstant
 *   <li>Reductions on ScalarConstant → ScalarConstant (value × reduction size or identity)
 *   <li>Reductions on IotaConstant → ScalarConstant (using arithmetic formulas)
 * </ul>
 *
 * <p>For example:
 *
 * <ul>
 *   <li>{@code sum(scalar(2.0).broadcast(3, 4))} → {@code scalar(24.0)} (2.0 × 12)
 *   <li>{@code sum(iota(10))} → {@code scalar(45.0)} (0+1+2+...+9 = 45)
 *   <li>{@code scalar(2.0) * scalar(3.0)} → {@code scalar(6.0)}
 * </ul>
 */
public final class TIRConstantFoldingPass implements TIRPass {

    @Override
    public TIRGraph run(TIRGraph graph) {
        // Collect the underlying ScalarConstants from graph inputs
        // Graph inputs may be wrapped in ViewTransforms, but the underlying
        // ScalarConstants represent runtime parameters and should NOT be folded
        Set<ScalarConstant> inputScalars = new HashSet<>();
        for (TIRNode input : graph.inputs()) {
            extractUnderlyingScalars(input, inputScalars);
        }
        return new ConstantFoldingRewriter(inputScalars).rewrite(graph);
    }

    /**
     * Extracts underlying ScalarConstant nodes from a graph input. ViewTransforms are traversed to
     * find the base scalar.
     */
    private void extractUnderlyingScalars(TIRNode node, Set<ScalarConstant> scalars) {
        if (node instanceof ScalarConstant sc) {
            scalars.add(sc);
        } else if (node instanceof ViewTransform vt) {
            extractUnderlyingScalars(vt.input(), scalars);
        }
        // TensorInput and other nodes don't contain ScalarConstants
    }

    @Override
    public String name() {
        return "ConstantFolding";
    }

    private static final class ConstantFoldingRewriter extends TIRRewriter {

        /**
         * ScalarConstants that are graph inputs - represent runtime parameters and must not be
         * folded.
         */
        private final Set<ScalarConstant> inputScalars;

        ConstantFoldingRewriter(Set<ScalarConstant> inputScalars) {
            this.inputScalars = inputScalars;
        }

        /**
         * Returns true if this ScalarConstant can be folded (compile-time constant). Graph inputs
         * are runtime parameters and cannot be folded.
         */
        private boolean canFold(ScalarConstant sc) {
            return !inputScalars.contains(sc);
        }

        @Override
        public TIRNode visitBinaryOp(BinaryOp node) {
            // First visit children
            TIRNode newLeft = rewriteChild(node.left());
            TIRNode newRight = rewriteChild(node.right());

            // Try to fold if both are scalar constants (and neither is a graph input)
            if (newLeft instanceof ScalarConstant left
                    && newRight instanceof ScalarConstant right
                    && canFold(left)
                    && canFold(right)) {
                ScalarConstant folded = foldBinary(node.op(), left, right);
                if (folded != null) {
                    return folded;
                }
            }

            if (newLeft == node.left() && newRight == node.right()) {
                return node;
            }
            return new BinaryOp(node.op(), newLeft, newRight, node.shape());
        }

        @Override
        public TIRNode visitUnaryOp(UnaryOp node) {
            TIRNode newInput = rewriteChild(node.input());

            // Try to fold if input is scalar constant (and not a graph input)
            if (newInput instanceof ScalarConstant input && canFold(input)) {
                ScalarConstant folded = foldUnary(node.op(), input);
                if (folded != null) {
                    return folded;
                }
            }

            if (newInput == node.input()) {
                return node;
            }
            return new UnaryOp(node.op(), newInput, node.shape());
        }

        @Override
        public TIRNode visitReductionOp(ReductionOp node) {
            TIRNode newInput = rewriteChild(node.input());

            // Try to fold reduction on scalar constant (if not a graph input)
            if (newInput instanceof ScalarConstant scalar && canFold(scalar)) {
                ScalarConstant folded = foldReductionScalar(node.op(), scalar, node.axes());
                if (folded != null) {
                    Shape outputShape = node.shape();
                    return ScalarConstant.broadcast(
                            folded.rawBits(), folded.dataType(), outputShape);
                }
            }

            // Try to fold reduction on iota constant
            // Note: IotaConstant is always virtual (computed from index), so it can always be
            // folded
            if (newInput instanceof IotaConstant iota) {
                ScalarConstant folded = foldReductionIota(node.op(), iota, node.axes());
                if (folded != null) {
                    Shape outputShape = node.shape();
                    return ScalarConstant.broadcast(
                            folded.rawBits(), folded.dataType(), outputShape);
                }
            }

            // Try to fold matmul pattern: sum(multiply(broadcast(scalarA), broadcast(scalarB)),
            // axes)
            // Only fold if BOTH scalars are foldable (i.e., not graph inputs)
            if (node.op() == ReductionOperator.SUM
                    && newInput instanceof BinaryOp binOp
                    && binOp.op() == BinaryOperator.MULTIPLY) {
                Optional<ScalarConstant> leftOpt = extractFoldableScalarConstant(binOp.left());
                Optional<ScalarConstant> rightOpt = extractFoldableScalarConstant(binOp.right());

                if (leftOpt.isPresent() && rightOpt.isPresent()) {
                    ScalarConstant folded =
                            foldMatmul(leftOpt.get(), rightOpt.get(), binOp, node.axes());
                    if (folded != null) {
                        Shape outputShape = node.shape();
                        return ScalarConstant.broadcast(
                                folded.rawBits(), folded.dataType(), outputShape);
                    }
                }
            }

            if (newInput == node.input()) {
                return node;
            }
            return new ReductionOp(
                    node.op(),
                    newInput,
                    node.axes(),
                    node.keepDims(),
                    node.accumulatorType(),
                    node.shape());
        }

        // ==================== Binary Folding ====================

        private ScalarConstant foldBinary(
                BinaryOperator op, ScalarConstant left, ScalarConstant right) {
            DataType dtype = left.dataType();
            if (dtype != right.dataType()) {
                return null; // Don't fold mixed types
            }

            // Compute broadcast shape
            Shape leftShape = left.shape();
            Shape rightShape = right.shape();
            Shape resultShape = broadcastShapes(leftShape, rightShape);

            long resultBits = evaluateBinary(op, left.rawBits(), right.rawBits(), dtype);
            return ScalarConstant.broadcast(resultBits, dtype, resultShape);
        }

        private long evaluateBinary(
                BinaryOperator op, long leftBits, long rightBits, DataType dtype) {
            if (dtype == DataType.FP32) {
                float a = Float.intBitsToFloat((int) leftBits);
                float b = Float.intBitsToFloat((int) rightBits);
                float result =
                        switch (op) {
                            case ADD -> a + b;
                            case SUBTRACT -> a - b;
                            case MULTIPLY -> a * b;
                            case DIVIDE -> a / b;
                            case MIN -> Math.min(a, b);
                            case MAX -> Math.max(a, b);
                            case POW -> (float) Math.pow(a, b);
                            default -> Float.NaN;
                        };
                return Float.floatToRawIntBits(result);
            } else if (dtype == DataType.FP64) {
                double a = Double.longBitsToDouble(leftBits);
                double b = Double.longBitsToDouble(rightBits);
                double result =
                        switch (op) {
                            case ADD -> a + b;
                            case SUBTRACT -> a - b;
                            case MULTIPLY -> a * b;
                            case DIVIDE -> a / b;
                            case MIN -> Math.min(a, b);
                            case MAX -> Math.max(a, b);
                            case POW -> Math.pow(a, b);
                            default -> Double.NaN;
                        };
                return Double.doubleToRawLongBits(result);
            } else if (dtype == DataType.I32) {
                int a = (int) leftBits;
                int b = (int) rightBits;
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
                            case SHIFT_LEFT -> a << (b & 31);
                            case SHIFT_RIGHT -> a >> (b & 31);
                            case SHIFT_RIGHT_UNSIGNED -> a >>> (b & 31);
                            default -> 0;
                        };
                return result;
            } else if (dtype == DataType.I64) {
                long a = leftBits;
                long b = rightBits;
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
                            case SHIFT_LEFT -> a << (b & 63);
                            case SHIFT_RIGHT -> a >> (b & 63);
                            case SHIFT_RIGHT_UNSIGNED -> a >>> (b & 63);
                            default -> 0;
                        };
                return result;
            }
            return 0;
        }

        // ==================== Unary Folding ====================

        private ScalarConstant foldUnary(UnaryOperator op, ScalarConstant input) {
            DataType dtype = input.dataType();
            long resultBits = evaluateUnary(op, input.rawBits(), dtype);
            return new ScalarConstant(resultBits, dtype, input.shape());
        }

        private long evaluateUnary(UnaryOperator op, long bits, DataType dtype) {
            if (dtype == DataType.FP32) {
                float a = Float.intBitsToFloat((int) bits);
                float result =
                        switch (op) {
                            case NEGATE -> -a;
                            case ABS -> Math.abs(a);
                            case SQRT -> (float) Math.sqrt(a);
                            case EXP -> (float) Math.exp(a);
                            case LOG -> (float) Math.log(a);
                            case SIN -> (float) Math.sin(a);
                            case COS -> (float) Math.cos(a);
                            case TAN -> (float) Math.tan(a);
                            case TANH -> (float) Math.tanh(a);
                            case RECIPROCAL -> 1.0f / a;
                            case LOGICAL_NOT -> a == 0 ? 1.0f : 0.0f;
                            case BITWISE_NOT -> Float.intBitsToFloat(~Float.floatToRawIntBits(a));
                        };
                return Float.floatToRawIntBits(result);
            } else if (dtype == DataType.FP64) {
                double a = Double.longBitsToDouble(bits);
                double result =
                        switch (op) {
                            case NEGATE -> -a;
                            case ABS -> Math.abs(a);
                            case SQRT -> Math.sqrt(a);
                            case EXP -> Math.exp(a);
                            case LOG -> Math.log(a);
                            case SIN -> Math.sin(a);
                            case COS -> Math.cos(a);
                            case TAN -> Math.tan(a);
                            case TANH -> Math.tanh(a);
                            case RECIPROCAL -> 1.0 / a;
                            case LOGICAL_NOT -> a == 0 ? 1.0 : 0.0;
                            case BITWISE_NOT ->
                                    Double.longBitsToDouble(~Double.doubleToRawLongBits(a));
                        };
                return Double.doubleToRawLongBits(result);
            } else if (dtype == DataType.I32) {
                int a = (int) bits;
                int result =
                        switch (op) {
                            case NEGATE -> -a;
                            case ABS -> Math.abs(a);
                            case LOGICAL_NOT -> a == 0 ? 1 : 0;
                            case BITWISE_NOT -> ~a;
                            default -> a;
                        };
                return result;
            } else if (dtype == DataType.I64) {
                long a = bits;
                long result =
                        switch (op) {
                            case NEGATE -> -a;
                            case ABS -> Math.abs(a);
                            case LOGICAL_NOT -> a == 0 ? 1 : 0;
                            case BITWISE_NOT -> ~a;
                            default -> a;
                        };
                return result;
            }
            return bits;
        }

        // ==================== Reduction Folding ====================

        private ScalarConstant foldReductionScalar(
                ReductionOperator op, ScalarConstant scalar, int[] axes) {
            // Compute the size of the reduced dimensions
            Shape shape = scalar.shape();
            long reducedSize = 1;
            for (int axis : axes) {
                reducedSize *= shape.flatAt(axis);
            }

            DataType dtype = scalar.dataType();
            long valueBits = scalar.rawBits();

            long resultBits =
                    switch (op) {
                        case SUM -> multiplyByCount(valueBits, reducedSize, dtype);
                        case PROD -> powerByCount(valueBits, reducedSize, dtype);
                        case MIN, MAX ->
                                valueBits; // min/max of identical values is the value itself
                    };

            return ScalarConstant.of(resultBits, dtype);
        }

        private ScalarConstant foldReductionIota(
                ReductionOperator op, IotaConstant iota, int[] axes) {
            // For iota, we can compute closed-form solutions
            long n = iota.count();
            DataType dtype = iota.dataType();

            // Only handle full reduction for now (all axes)
            Shape shape = iota.shape();
            long totalSize = shape.size();
            if (n != totalSize) {
                return null; // Partial reduction on reshaped iota is complex
            }

            // Check if reducing all dimensions
            long reducedSize = 1;
            for (int axis : axes) {
                reducedSize *= shape.flatAt(axis);
            }
            if (reducedSize != totalSize) {
                return null; // Partial reduction
            }

            long resultBits =
                    switch (op) {
                        case SUM -> {
                            // sum(0..n-1) = n*(n-1)/2
                            long sum = n * (n - 1) / 2;
                            yield toLongBits(sum, dtype);
                        }
                        case PROD -> {
                            // prod(0..n-1) = 0 if n > 0 (because 0 is included)
                            yield toLongBits(0, dtype);
                        }
                        case MIN -> {
                            // min(0..n-1) = 0
                            yield toLongBits(0, dtype);
                        }
                        case MAX -> {
                            // max(0..n-1) = n-1
                            yield toLongBits(n - 1, dtype);
                        }
                    };

            return ScalarConstant.of(resultBits, dtype);
        }

        // ==================== Helper Methods ====================

        private long multiplyByCount(long valueBits, long count, DataType dtype) {
            if (dtype == DataType.FP32) {
                float value = Float.intBitsToFloat((int) valueBits);
                return Float.floatToRawIntBits(value * count);
            } else if (dtype == DataType.FP64) {
                double value = Double.longBitsToDouble(valueBits);
                return Double.doubleToRawLongBits(value * count);
            } else if (dtype == DataType.I32) {
                return (int) valueBits * count;
            } else if (dtype == DataType.I64) {
                return valueBits * count;
            }
            return valueBits;
        }

        private long powerByCount(long valueBits, long count, DataType dtype) {
            if (dtype == DataType.FP32) {
                float value = Float.intBitsToFloat((int) valueBits);
                return Float.floatToRawIntBits((float) Math.pow(value, count));
            } else if (dtype == DataType.FP64) {
                double value = Double.longBitsToDouble(valueBits);
                return Double.doubleToRawLongBits(Math.pow(value, count));
            } else if (dtype == DataType.I32) {
                return (long) Math.pow((int) valueBits, count);
            } else if (dtype == DataType.I64) {
                return (long) Math.pow(valueBits, count);
            }
            return valueBits;
        }

        private long toLongBits(long value, DataType dtype) {
            if (dtype == DataType.FP32) {
                return Float.floatToRawIntBits((float) value);
            } else if (dtype == DataType.FP64) {
                return Double.doubleToRawLongBits((double) value);
            } else {
                return value;
            }
        }

        private Shape broadcastShapes(Shape a, Shape b) {
            int rankA = a.flatRank();
            int rankB = b.flatRank();
            int maxRank = Math.max(rankA, rankB);

            long[] result = new long[maxRank];
            for (int i = 0; i < maxRank; i++) {
                long dimA = (i < rankA) ? a.flatAt(rankA - 1 - i) : 1;
                long dimB = (i < rankB) ? b.flatAt(rankB - 1 - i) : 1;

                if (dimA == dimB) {
                    result[maxRank - 1 - i] = dimA;
                } else if (dimA == 1) {
                    result[maxRank - 1 - i] = dimB;
                } else if (dimB == 1) {
                    result[maxRank - 1 - i] = dimA;
                } else {
                    throw new IllegalArgumentException(
                            "Shapes not broadcastable: " + a + " and " + b);
                }
            }
            return Shape.flat(result);
        }

        // ==================== Matmul Folding ====================

        /**
         * Extracts a ScalarConstant from a node. This handles: 1. Direct ScalarConstant nodes 2.
         * ScalarConstants wrapped in ViewTransform chains (broadcast, expand, reshape) 3.
         * TensorInput nodes representing broadcasted scalars (all-zero strides with single element)
         *
         * <p>In all these cases, all elements of the tensor have the same constant value.
         *
         * @param node the TIR node to examine
         * @return Optional containing the ScalarConstant if found, empty otherwise
         */
        private Optional<ScalarConstant> extractScalarConstant(TIRNode node) {
            // Case 1: Direct ScalarConstant
            if (node instanceof ScalarConstant sc) {
                return Optional.of(sc);
            }

            // Case 2: ViewTransform wrapping a scalar constant
            if (node instanceof ViewTransform vt) {
                return extractFromViewTransform(vt);
            }

            // Case 3: TensorInput with broadcasted scalar layout (all-zero strides)
            if (node instanceof TensorInput ti) {
                return extractFromTensorInput(ti);
            }

            return Optional.empty();
        }

        /**
         * Extracts a ScalarConstant from a node, but only if it can be folded (i.e., it's not a
         * graph input). Graph inputs represent runtime parameters.
         *
         * @param node the TIR node to examine
         * @return Optional containing the foldable ScalarConstant, empty otherwise
         */
        private Optional<ScalarConstant> extractFoldableScalarConstant(TIRNode node) {
            Optional<ScalarConstant> sc = extractScalarConstant(node);
            // Only return if the scalar is not a graph input
            return sc.filter(this::canFold);
        }

        /**
         * Extracts scalar constant from a ViewTransform chain. Follows through broadcasts, expands,
         * and reshapes to find the underlying scalar.
         */
        private Optional<ScalarConstant> extractFromViewTransform(ViewTransform vt) {
            // Check if this view preserves the "scalar constant" property
            if (vt.kind() instanceof ViewKind.Broadcast b) {
                // If broadcasting from a scalar, follow the chain
                if (b.fromShape().isScalar()) {
                    return extractScalarConstant(vt.input());
                }
                // Check if input is itself a scalar broadcast
                Optional<ScalarConstant> inputSc = extractScalarConstant(vt.input());
                if (inputSc.isPresent()) {
                    return inputSc;
                }
            } else if (vt.kind() instanceof ViewKind.Expand e) {
                if (e.fromShape().isScalar()) {
                    return extractScalarConstant(vt.input());
                }
                Optional<ScalarConstant> inputSc = extractScalarConstant(vt.input());
                if (inputSc.isPresent()) {
                    return inputSc;
                }
            } else if (vt.kind() instanceof ViewKind.Reshape) {
                // Reshape preserves values
                return extractScalarConstant(vt.input());
            } else if (vt.kind() instanceof ViewKind.Transpose) {
                // Transpose preserves values
                return extractScalarConstant(vt.input());
            }
            // Slice breaks the constant property
            return Optional.empty();
        }

        /**
         * Extracts scalar constant from a TensorInput. A TensorInput represents a broadcasted
         * scalar if it has all-zero strides (meaning all elements point to the same memory
         * location).
         */
        private Optional<ScalarConstant> extractFromTensorInput(TensorInput ti) {
            // Check if this is a broadcasted scalar by examining strides
            // All-zero strides means it's a broadcast of a single value
            boolean allZeroStrides = true;
            long[] strides = ti.layout().stride().toArray();
            for (long stride : strides) {
                if (stride != 0) {
                    allZeroStrides = false;
                    break;
                }
            }

            if (allZeroStrides) {
                // This is a broadcasted value - we need to extract the actual value
                // However, TensorInput doesn't store the value directly; it's a placeholder
                // for a runtime input. We can't constant-fold this without knowing the value.
                // Return empty to indicate we can't fold this case.
                return Optional.empty();
            }

            return Optional.empty();
        }

        /**
         * Recursive helper that tracks whether we're still in a "scalar origin" chain.
         *
         * @param node the TIR node to examine
         * @param fromScalar whether the chain so far has originated from a scalar
         * @return Optional containing the ScalarConstant if the chain ends at one, empty otherwise
         */
        private Optional<ScalarConstant> extractScalarConstantRecursive(
                TIRNode node, boolean fromScalar) {
            if (node instanceof ScalarConstant sc) {
                // Only return the scalar if the entire chain originated from a scalar
                return fromScalar ? Optional.of(sc) : Optional.empty();
            }
            if (node instanceof ViewTransform vt) {
                // Check if this view preserves the "constant" property
                boolean stillFromScalar = fromScalar;

                if (vt.kind() instanceof ViewKind.Broadcast b) {
                    // If broadcasting from a scalar, continue tracking
                    stillFromScalar = fromScalar && b.fromShape().isScalar();
                    // Even if fromShape isn't scalar, if we came from a scalar and this is
                    // broadcasting a size-1 dim, it might still be constant
                    if (!stillFromScalar && fromScalar) {
                        // Check if all elements in fromShape would be the same constant
                        // This happens when the input is itself a scalar broadcast
                        Optional<ScalarConstant> inputSc =
                                extractScalarConstantRecursive(vt.input(), true);
                        if (inputSc.isPresent()) {
                            return inputSc;
                        }
                    }
                } else if (vt.kind() instanceof ViewKind.Expand e) {
                    // Expand from scalar preserves constant property
                    stillFromScalar = fromScalar && e.fromShape().isScalar();
                    if (!stillFromScalar && fromScalar) {
                        Optional<ScalarConstant> inputSc =
                                extractScalarConstantRecursive(vt.input(), true);
                        if (inputSc.isPresent()) {
                            return inputSc;
                        }
                    }
                } else if (vt.kind() instanceof ViewKind.Reshape r) {
                    // Reshape preserves values, so if input was scalar-originated, output is too
                    // but we need to check if input was actually a scalar broadcast
                    if (fromScalar) {
                        Optional<ScalarConstant> inputSc =
                                extractScalarConstantRecursive(vt.input(), true);
                        if (inputSc.isPresent()) {
                            return inputSc;
                        }
                    }
                    stillFromScalar = false;
                } else if (vt.kind() instanceof ViewKind.Slice) {
                    // Slice breaks the constant broadcast property
                    stillFromScalar = false;
                }

                if (stillFromScalar) {
                    return extractScalarConstantRecursive(vt.input(), true);
                }
            }
            return Optional.empty();
        }

        /**
         * Folds a matmul operation when both inputs are broadcasted scalar constants.
         *
         * <p>Matmul of two constant-filled tensors: if A is filled with value 'a' and B is filled
         * with value 'b', then (A @ B)[i,j] = sum_k(A[i,k] * B[k,j]) = sum_k(a * b) = K * a * b
         * where K is the contraction dimension size.
         *
         * @param left the left scalar constant
         * @param right the right scalar constant
         * @param mulOp the multiply binary operation (for computing the broadcast shape)
         * @param axes the reduction axes (contraction dimensions)
         * @return the folded ScalarConstant, or null if types don't match
         */
        private ScalarConstant foldMatmul(
                ScalarConstant left, ScalarConstant right, BinaryOp mulOp, int[] axes) {
            DataType dtype = left.dataType();
            if (dtype != right.dataType()) {
                return null; // Don't fold mixed types
            }

            // Compute the contraction dimension size (K)
            long contractionSize = 1;
            Shape mulShape = mulOp.shape();
            for (int axis : axes) {
                if (axis >= 0 && axis < mulShape.rank()) {
                    contractionSize *= mulShape.flatAt(axis);
                }
            }

            long resultBits =
                    evaluateMatmul(left.rawBits(), right.rawBits(), contractionSize, dtype);
            return ScalarConstant.of(resultBits, dtype);
        }

        /**
         * Evaluates a matmul of two scalar values over a contraction dimension.
         *
         * <p>Result = contractionSize * left * right
         *
         * @param leftBits raw bits of left scalar value
         * @param rightBits raw bits of right scalar value
         * @param contractionSize size of the contraction dimension (K)
         * @param dtype the data type
         * @return raw bits of the result
         */
        private long evaluateMatmul(
                long leftBits, long rightBits, long contractionSize, DataType dtype) {
            if (dtype == DataType.FP32) {
                float left = Float.intBitsToFloat((int) leftBits);
                float right = Float.intBitsToFloat((int) rightBits);
                float result = left * right * contractionSize;
                return Float.floatToRawIntBits(result);
            } else if (dtype == DataType.FP64) {
                double left = Double.longBitsToDouble(leftBits);
                double right = Double.longBitsToDouble(rightBits);
                double result = left * right * contractionSize;
                return Double.doubleToRawLongBits(result);
            } else if (dtype == DataType.I32) {
                return (int) leftBits * (int) rightBits * contractionSize;
            } else if (dtype == DataType.I64) {
                return leftBits * rightBits * contractionSize;
            }
            return 0;
        }
    }
}
