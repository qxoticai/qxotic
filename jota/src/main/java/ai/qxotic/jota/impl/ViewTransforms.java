package ai.qxotic.jota.impl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.Util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class ViewTransforms {

    private ViewTransforms() {}

    public record ViewTransformSpec(Layout layout, long byteOffsetDelta) {}

    public static ViewTransformSpec view(Layout layout, Shape newShape) {
        if (layout.shape().size() != newShape.size()) {
            throw new IllegalArgumentException("total element count mismatch");
        }

        if (spansContiguousRange(layout)) {
            return new ViewTransformSpec(Layout.rowMajor(newShape), 0L);
        }

        long[] oldStrides = layout.stride().toArray();
        if (!canReshapeWithoutCopy(layout.shape(), newShape, oldStrides)) {
            throw new IllegalArgumentException("Cannot view: would require copying data");
        }

        long[] newStrides = computeReshapeStrides(layout.shape(), newShape, oldStrides);
        Layout newLayout = Layout.of(newShape, Stride.template(newShape, newStrides));
        return new ViewTransformSpec(newLayout, 0L);
    }

    public static ViewTransformSpec expand(Layout layout, Shape newShape) {
        Shape currentShape = layout.shape();

        if (!currentShape.isCongruentWith(newShape)) {
            throw new IllegalArgumentException(
                    "expand requires congruent shapes: current="
                            + currentShape
                            + ", target="
                            + newShape);
        }

        long[] currentStrides = layout.stride().toArray();
        long[] newStrides = new long[currentShape.flatRank()];

        for (int i = 0; i < currentShape.flatRank(); i++) {
            long currentDim = currentShape.flatAt(i);
            long newDim = newShape.flatAt(i);

            if (currentDim == 1) {
                newStrides[i] = 0;
            } else if (currentDim == newDim) {
                newStrides[i] = currentStrides[i];
            } else {
                throw new IllegalArgumentException(
                        "Cannot expand dimension "
                                + i
                                + " from size "
                                + currentDim
                                + " to "
                                + newDim);
            }
        }

        Stride newStride = Stride.template(newShape, newStrides);
        Layout newLayout = Layout.of(newShape, newStride);
        return new ViewTransformSpec(newLayout, 0L);
    }

    public static ViewTransformSpec broadcast(Layout layout, Shape targetShape) {
        Shape currentShape = layout.shape();
        int numNewModes = targetShape.rank() - currentShape.rank();
        if (numNewModes < 0) {
            throw new IllegalArgumentException(
                    "Cannot broadcast shape "
                            + currentShape
                            + " to shape "
                            + targetShape
                            + ": target has fewer modes");
        }

        if (numNewModes == 0) {
            return expand(layout, targetShape);
        }

        if (!currentShape.isFlat() || !targetShape.isFlat()) {
            long[] newDims = new long[targetShape.flatRank()];
            long[] currentDims = currentShape.toArray();

            int prepend = targetShape.flatRank() - currentShape.flatRank();
            Arrays.fill(newDims, 0, prepend, 1);
            System.arraycopy(currentDims, 0, newDims, prepend, currentDims.length);

            Shape reshapedShape = Shape.flat(newDims);
            ViewTransformSpec reshaped = view(layout, reshapedShape);
            return expand(reshaped.layout(), targetShape);
        }

        long[] newDims = new long[targetShape.rank()];
        for (int i = 0; i < numNewModes; i++) {
            newDims[i] = 1;
        }
        System.arraycopy(currentShape.toArray(), 0, newDims, numNewModes, currentShape.rank());

        ViewTransformSpec reshaped = view(layout, Shape.flat(newDims));
        return expand(reshaped.layout(), targetShape);
    }

    public static ViewTransformSpec permute(Layout layout, int... permutationIndices) {
        Shape newShape = layout.shape().permute(permutationIndices);
        Stride newStride = layout.stride().permute(permutationIndices);
        Layout newLayout = Layout.of(newShape, newStride);
        return new ViewTransformSpec(newLayout, 0L);
    }

    public static ViewTransformSpec transpose(Layout layout, int _axis0, int _axis1) {
        Shape shape = layout.shape();
        int axis0 = Util.wrapAround(_axis0, shape.rank());
        int axis1 = Util.wrapAround(_axis1, shape.rank());
        int[] permutation = new int[shape.rank()];
        for (int i = 0; i < shape.rank(); i++) {
            permutation[i] = i;
        }
        permutation[axis0] = axis1;
        permutation[axis1] = axis0;
        return permute(layout, permutation);
    }

    public static ViewTransformSpec slice(
            Layout layout,
            DataType dataType,
            int _axis,
            long fromInclusive,
            long toExclusive,
            long indexStride) {
        int axis = Util.wrapAround(_axis, layout.shape().rank());
        long dimSize = layout.shape().size(axis);

        if (indexStride == 0) {
            throw new IllegalArgumentException("Step cannot be zero");
        }

        if (indexStride > 0) {
            if (fromInclusive < 0 || toExclusive > dimSize || fromInclusive > toExclusive) {
                throw new IllegalArgumentException(
                        String.format(
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
                                fromInclusive, toExclusive, indexStride, axis, dimSize));
            }
        } else {
            if (fromInclusive < 0
                    || fromInclusive >= dimSize
                    || toExclusive < -1
                    || toExclusive >= dimSize
                    || fromInclusive < toExclusive) {
                throw new IllegalArgumentException(
                        String.format(
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
                                fromInclusive, toExclusive, indexStride, axis, dimSize));
            }
        }

        long byteStride = layout.stride().modeAt(axis).flatAt(0) * dataType.byteSize();
        long byteOffsetDelta = fromInclusive * byteStride;

        long length;
        if (indexStride > 0) {
            length = (toExclusive - fromInclusive + indexStride - 1) / indexStride;
        } else {
            length = (toExclusive - fromInclusive + indexStride + 1) / indexStride;
        }
        if (length < 0) {
            length = 0;
        }

        Shape newModeShape = Shape.flat(length);
        Shape newShape = layout.shape().replace(axis, newModeShape);
        Stride newModeStride = layout.stride().modeAt(axis).scale(indexStride);
        Stride newStride = layout.stride().replace(axis, newModeStride);

        Layout newLayout = Layout.of(newShape, newStride);
        return new ViewTransformSpec(newLayout, byteOffsetDelta);
    }

    private static boolean spansContiguousRange(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return true;
        }
        long span = 0;
        long totalElements = 1;
        long[] strides = layout.stride().toArray();
        for (int i = 0; i < layout.shape().flatRank(); i++) {
            long dim = layout.shape().flatAt(i);
            span += (dim - 1) * strides[i];
            totalElements *= dim;
        }
        return span == totalElements - 1;
    }

    private static boolean canReshapeWithoutCopy(
            Shape oldShape, Shape newShape, long[] oldStrides) {
        List<Long> oldDims = new ArrayList<>();
        List<Long> oldStridesFiltered = new ArrayList<>();
        for (int i = 0; i < oldShape.flatRank(); i++) {
            long dim = oldShape.flatAt(i);
            if (dim != 1) {
                oldDims.add(dim);
                oldStridesFiltered.add(oldStrides[i]);
            }
        }

        List<Long> newDims = new ArrayList<>();
        for (int i = 0; i < newShape.flatRank(); i++) {
            long dim = newShape.flatAt(i);
            if (dim != 1) {
                newDims.add(dim);
            }
        }

        if (oldDims.equals(newDims)) {
            return true;
        }

        return canGroupDimensions(oldDims, newDims, oldStridesFiltered);
    }

    private static boolean canGroupDimensions(
            List<Long> oldDims, List<Long> newDims, List<Long> oldStrides) {
        if (oldDims.isEmpty() && newDims.isEmpty()) {
            return true;
        }

        int oldIdx = 0;
        int newIdx = 0;
        while (oldIdx < oldDims.size() && newIdx < newDims.size()) {
            long oldProduct = 1;
            long newProduct = 1;
            int oldStart = oldIdx;

            while (oldIdx < oldDims.size() && oldProduct < newDims.get(newIdx)) {
                oldProduct *= oldDims.get(oldIdx);
                oldIdx++;
            }

            while (newIdx < newDims.size() && newProduct < oldProduct) {
                newProduct *= newDims.get(newIdx);
                newIdx++;
            }

            if (oldProduct != newProduct) {
                return false;
            }

            if (!areContiguous(oldDims, oldStrides, oldStart, oldIdx)) {
                return false;
            }
        }

        return oldIdx == oldDims.size() && newIdx == newDims.size();
    }

    private static boolean areContiguous(
            List<Long> dims, List<Long> strides, int startIdx, int endIdx) {
        if (endIdx - startIdx == 0) {
            return true;
        }

        long span = 0;
        long totalElements = 1;
        for (int i = startIdx; i < endIdx; i++) {
            span += (dims.get(i) - 1) * strides.get(i);
            totalElements *= dims.get(i);
        }

        return span == totalElements - 1;
    }

    private static long[] computeReshapeStrides(Shape oldShape, Shape newShape, long[] oldStrides) {
        List<Long> oldDimsNonSingleton = new ArrayList<>();
        List<Long> oldStridesNonSingleton = new ArrayList<>();
        for (int i = 0; i < oldShape.flatRank(); i++) {
            if (oldShape.flatAt(i) != 1) {
                oldDimsNonSingleton.add(oldShape.flatAt(i));
                oldStridesNonSingleton.add(oldStrides[i]);
            }
        }

        List<Long> newDimsNonSingleton = new ArrayList<>();
        List<Integer> newDimIndices = new ArrayList<>();
        for (int i = 0; i < newShape.flatRank(); i++) {
            if (newShape.flatAt(i) != 1) {
                newDimsNonSingleton.add(newShape.flatAt(i));
                newDimIndices.add(i);
            }
        }

        long[] newStrides = new long[newShape.flatRank()];

        if (oldDimsNonSingleton.equals(newDimsNonSingleton)) {
            int k = 0;
            for (int i = 0; i < newShape.flatRank(); i++) {
                if (newShape.flatAt(i) != 1) {
                    newStrides[i] = oldStridesNonSingleton.get(k++);
                }
            }
            for (int i = newShape.flatRank() - 1; i >= 0; i--) {
                if (newShape.flatAt(i) == 1) {
                    newStrides[i] = (i == newShape.flatRank() - 1) ? 1 : newStrides[i + 1];
                }
            }
            return newStrides;
        }

        int oldIdx = 0;
        int newIdx = 0;
        while (oldIdx < oldDimsNonSingleton.size() && newIdx < newDimsNonSingleton.size()) {
            long oldProduct = 1;
            long newProduct = 1;
            int oldStart = oldIdx;
            int newStart = newIdx;

            while (oldIdx < oldDimsNonSingleton.size()
                    && oldProduct < newDimsNonSingleton.get(newIdx)) {
                oldProduct *= oldDimsNonSingleton.get(oldIdx++);
            }
            while (newIdx < newDimsNonSingleton.size() && newProduct < oldProduct) {
                newProduct *= newDimsNonSingleton.get(newIdx++);
            }
            if (oldProduct != newProduct) {
                throw new IllegalArgumentException(
                        "Cannot reshape: incompatible dimension grouping");
            }

            long baseStride = oldStridesNonSingleton.get(oldStart);
            for (int i = newStart; i < newIdx; i++) {
                int actualNewIdx = newDimIndices.get(i);
                if (i == newIdx - 1) {
                    newStrides[actualNewIdx] = baseStride;
                } else {
                    long stride = baseStride;
                    for (int j = i + 1; j < newIdx; j++) {
                        int laterIdx = newDimIndices.get(j);
                        stride *= newShape.flatAt(laterIdx);
                    }
                    newStrides[actualNewIdx] = stride;
                }
            }
        }

        for (int i = newShape.flatRank() - 1; i >= 0; i--) {
            if (newStrides[i] == 0) {
                newStrides[i] = (i == newShape.flatRank() - 1) ? 1 : newStrides[i + 1];
            }
        }

        return newStrides;
    }
}
