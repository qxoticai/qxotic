package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/**
 * Gather operation node in IR-T.
 *
 * <p>Gathers elements from the input tensor along the specified axis according to the indices. This
 * is commonly used for embedding lookups and index-based tensor access.
 *
 * <p>Output shape: input.shape with `axis` dimension replaced by indices.shape
 *
 * <p>Example: - input: [N, C, H, W] with axis=1 - indices: [N, K, H, W] - output: [N, K, H, W]
 */
public record GatherOp(TIRNode input, TIRNode indices, int axis, Shape shape) implements TIRNode {

    public GatherOp {
        Objects.requireNonNull(input);
        Objects.requireNonNull(indices);
        Objects.requireNonNull(shape);

        // Validate axis is within bounds
        int inputRank = input.shape().rank();
        int normalizedAxis = axis < 0 ? axis + inputRank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= inputRank) {
            throw new IllegalArgumentException(
                    "Gather axis " + axis + " is out of bounds for input rank " + inputRank);
        }

        // Validate indices data type is integral
        DataType indicesType = indices.dataType();
        if (!indicesType.isIntegral()) {
            throw new IllegalArgumentException(
                    "Gather indices must be integral type, got " + indicesType);
        }
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Shape shape() {
        return shape;
    }

    /**
     * Computes the output shape for a gather operation.
     *
     * <p>The output shape is constructed as: - input dimensions before axis - indices dimensions -
     * input dimensions after axis
     *
     * <p>Example: - input: [3, 4, 5], indices: [2, 6], axis: 1 - output: [3, 2, 6, 5]
     *
     * @param inputShape shape of the input tensor
     * @param indicesShape shape of the indices tensor
     * @param axis axis along which to gather
     * @return the output shape
     */
    public static Shape computeOutputShape(Shape inputShape, Shape indicesShape, int axis) {
        int inputRank = inputShape.rank();
        int indicesRank = indicesShape.rank();

        // Normalize axis
        int normalizedAxis = axis < 0 ? axis + inputRank : axis;

        // Output rank = inputRank - 1 (remove gathered axis) + indicesRank (add indices dimensions)
        int outputRank = inputRank - 1 + indicesRank;
        long[] outputDims = new long[outputRank];

        int outIdx = 0;

        // Copy input dimensions before axis
        for (int i = 0; i < normalizedAxis; i++) {
            outputDims[outIdx++] = inputShape.flatAt(i);
        }

        // Copy indices dimensions
        for (int i = 0; i < indicesRank; i++) {
            outputDims[outIdx++] = indicesShape.flatAt(i);
        }

        // Copy input dimensions after axis
        for (int i = normalizedAxis + 1; i < inputRank; i++) {
            outputDims[outIdx++] = inputShape.flatAt(i);
        }

        return Shape.flat(outputDims);
    }
}
