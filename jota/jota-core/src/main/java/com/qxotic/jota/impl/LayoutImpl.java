package com.qxotic.jota.impl;

import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.Util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

final class LayoutImpl implements Layout {

    private static final Layout SCALAR = LayoutImpl.of(Shape.scalar(), Stride.scalar());

    private final Shape shape;
    private final Stride stride;

    private LayoutImpl(Shape shape, Stride stride) {
        Objects.requireNonNull(shape);
        Objects.requireNonNull(stride);

        if (shape.flatRank() != stride.flatRank()) {
            throw new IllegalArgumentException(
                    "Shape and stride must have the same flatRank: "
                            + "shape.flatRank()="
                            + shape.flatRank()
                            + " stride.flatRank()="
                            + stride.flatRank()
                            + " (shape="
                            + shape
                            + " stride="
                            + stride
                            + ")");
        }

        // Note: We allow different nesting structures as long as flatRank matches,
        // following CuTe's design. Use isCongruent() to check structural equivalence.
        assert shape.isCongruentWith(stride);
        this.shape = shape;
        this.stride = stride;
    }

    public static Layout scalar() {
        return SCALAR;
    }

    @Override
    public Shape shape() {
        return this.shape;
    }

    @Override
    public Stride stride() {
        return this.stride;
    }

    @Override
    public Layout modeAt(int modeIndex) {
        Shape modeShape = shape.modeAt(modeIndex);
        Stride modeStride = stride.modeAt(modeIndex);
        return new LayoutImpl(modeShape, modeStride);
    }

    @Override
    public Layout coalesce() {
        if (shape.flatRank() == 0) {
            return this;
        }

        Layout flat = flatten();
        long[] dims = flat.shape().toArray();
        long[] strides = flat.stride().toArray();
        long[] newDims = new long[dims.length];
        long[] newStrides = new long[strides.length];
        int count = 0;

        for (int i = 0; i < dims.length; i++) {
            long dim = dims[i];
            long strideValue = strides[i];

            if (dim == 1) {
                continue;
            }

            if (count == 0) {
                newDims[0] = dim;
                newStrides[0] = strideValue;
                count = 1;
                continue;
            }

            long prevDim = newDims[count - 1];
            long prevStride = newStrides[count - 1];

            long expectedPrevStride = Math.multiplyExact(dim, strideValue);

            if (prevStride == expectedPrevStride) {
                newDims[count - 1] = Math.multiplyExact(prevDim, dim);
                newStrides[count - 1] = strideValue;
            } else {
                newDims[count] = dim;
                newStrides[count] = strideValue;
                count++;
            }
        }

        if (count == 0) {
            return Layout.of(Shape.flat(1), Stride.flat(0));
        }

        if (count == dims.length) {
            return flat;
        }

        return Layout.of(
                Shape.flat(Arrays.copyOf(newDims, count)),
                Stride.flat(Arrays.copyOf(newStrides, count)));
    }

    @Override
    public boolean canCoalesce(int _modeIndex) {
        int rank = shape.rank();
        if (rank < 2) {
            return false;
        }

        int modeIndex = Util.wrapAround(_modeIndex, rank);
        if (modeIndex == rank - 1) {
            return false;
        }

        ModePair pair = modePairAt(modeIndex);
        return pair != null && canMerge(pair.leftStride, pair.rightDim, pair.rightStride);
    }

    @Override
    public Layout coalesce(int _modeIndex) {
        int rank = shape.rank();
        if (rank < 2) {
            return this;
        }

        int modeIndex = Util.wrapAround(_modeIndex, rank);
        if (modeIndex == rank - 1) {
            return this;
        }

        ModePair pair = modePairAt(modeIndex);
        if (pair == null || !canMerge(pair.leftStride, pair.rightDim, pair.rightStride)) {
            return this;
        }

        if (shape.isFlat() && stride.isFlat()) {
            return coalesceFlatPair(modeIndex, pair);
        }
        return coalesceNestedPair(modeIndex, pair);
    }

    private ModePair modePairAt(int modeIndex) {
        if (shape.isFlat() && stride.isFlat()) {
            return new ModePair(
                    shape.flatAt(modeIndex),
                    stride.flatAt(modeIndex),
                    shape.flatAt(modeIndex + 1),
                    stride.flatAt(modeIndex + 1));
        }

        Layout leftMode = modeAt(modeIndex).coalesce();
        Layout rightMode = modeAt(modeIndex + 1).coalesce();
        if (leftMode.shape().rank() != 1 || rightMode.shape().rank() != 1) {
            return null;
        }

        return new ModePair(
                leftMode.shape().flatAt(0),
                leftMode.stride().flatAt(0),
                rightMode.shape().flatAt(0),
                rightMode.stride().flatAt(0));
    }

    private Layout coalesceFlatPair(int modeIndex, ModePair pair) {
        long[] dims = shape.toArray();
        long[] strides = stride.toArray();
        long mergedDim = Math.multiplyExact(pair.leftDim, pair.rightDim);

        long[] newDims = new long[dims.length - 1];
        long[] newStrides = new long[strides.length - 1];
        System.arraycopy(dims, 0, newDims, 0, modeIndex);
        System.arraycopy(strides, 0, newStrides, 0, modeIndex);
        newDims[modeIndex] = mergedDim;
        newStrides[modeIndex] = pair.rightStride;
        System.arraycopy(dims, modeIndex + 2, newDims, modeIndex + 1, dims.length - modeIndex - 2);
        System.arraycopy(
                strides, modeIndex + 2, newStrides, modeIndex + 1, strides.length - modeIndex - 2);

        return Layout.of(Shape.flat(newDims), Stride.flat(newStrides));
    }

    private Layout coalesceNestedPair(int modeIndex, ModePair pair) {
        long mergedDim = Math.multiplyExact(pair.leftDim, pair.rightDim);

        Shape mergedShape = Shape.flat(mergedDim);
        Stride mergedStride = Stride.flat(pair.rightStride);
        Shape updatedShape = shape.replace(modeIndex, mergedShape).remove(modeIndex + 1);
        Stride updatedStride = stride.replace(modeIndex, mergedStride).remove(modeIndex + 1);
        return Layout.of(updatedShape, updatedStride);
    }

    @Override
    public boolean isCongruentWith(Layout other) {
        Objects.requireNonNull(other);
        return this.shape.isCongruentWith(other.shape())
                && this.stride.isCongruentWith(other.stride());
    }

    @Override
    public Layout compose(Layout inner) {
        Objects.requireNonNull(inner);
        return representAsAffineLayout(
                inner.shape(),
                coord -> composeOffset(inner, coord),
                "compose is non-representable as Layout(shape, stride) in the current model");
    }

    @Override
    public Layout inverse() {
        if (!isBijective()) {
            throw new IllegalArgumentException("inverse requires a bijective layout");
        }

        SpanInfo span = spanInfo(this);
        if (span.minOffset != 0 || span.maxOffset != this.shape.size() - 1) {
            throw new IllegalArgumentException(
                    "inverse requires a zero-based dense codomain [0, size-1]");
        }

        long[] inverseLinear = buildInverseLookup();
        return representAsAffineLayout(
                shape,
                coord -> inverseLinear[(int) Indexing.coordToLinear(shape, coord)],
                "inverse is non-representable as Layout(shape, stride) in the current model");
    }

    @Override
    public Layout complement(long cotarget) {
        if (cotarget <= 0) {
            throw new IllegalArgumentException("cotarget must be > 0");
        }
        if (shape.hasZeroElements()) {
            return Layout.rowMajor(cotarget);
        }

        Layout normalized = coalesce();
        if (!normalized.isInjective()) {
            throw new IllegalArgumentException("complement requires an injective layout");
        }

        List<Mode> modes = normalizedModes(normalized);
        if (modes.isEmpty()) {
            return Layout.rowMajor(cotarget);
        }

        for (Mode mode : modes) {
            if (mode.stride < 0) {
                throw new IllegalArgumentException(
                        "complement currently requires non-negative strides");
            }
        }

        modes.sort(Comparator.comparingLong(m -> m.stride));

        List<Long> complementDims = new ArrayList<>();
        List<Long> complementStrides = new ArrayList<>();
        long running = 1;

        for (Mode mode : modes) {
            if (mode.stride % running != 0) {
                throw new IllegalArgumentException(
                        "complement requires divisible stride chain in normalized layout");
            }

            long gap = mode.stride / running;
            if (gap > 1) {
                complementDims.add(gap);
                complementStrides.add(running);
            }

            running = Math.multiplyExact(running, gap);
            running = Math.multiplyExact(running, mode.dim);
        }

        long tail = ceilDiv(cotarget, running);
        if (tail > 1) {
            complementDims.add(tail);
            complementStrides.add(running);
        }

        if (complementDims.isEmpty()) {
            return Layout.scalar();
        }

        long[] dims = toArray(complementDims);
        long[] strides = toArray(complementStrides);
        return Layout.of(Shape.flat(dims), Stride.flat(strides)).coalesce();
    }

    private long composeOffset(Layout inner, long[] coord) {
        long innerLinear = evaluateOffset(inner, coord);
        validateComposeDomain(innerLinear);
        return evaluateOffset(this, Indexing.linearToCoord(shape, innerLinear));
    }

    private void validateComposeDomain(long innerLinear) {
        long outerSize = shape.size();
        if (innerLinear < 0 || innerLinear >= outerSize) {
            throw new IllegalArgumentException(
                    "compose domain mismatch: inner layout maps outside outer domain [0, "
                            + (outerSize - 1)
                            + "]");
        }
    }

    private long[] buildInverseLookup() {
        long size = shape.size();
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "inverse is only supported for layouts with size <= " + Integer.MAX_VALUE);
        }

        long[] inverseLinear = new long[(int) size];
        Arrays.fill(inverseLinear, -1L);
        fillInverseLookup(this, inverseLinear);
        return inverseLinear;
    }

    @Override
    public String toString() {
        return shape.toString() + ":" + stride.toString();
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof Layout that)
                && Objects.equals(this.shape, that.shape())
                && Objects.equals(this.stride, that.stride());
    }

    @Override
    public int hashCode() {
        return Objects.hash(shape, stride);
    }

    private static Layout representAsAffineLayout(
            Shape domain, CoordinateMap map, String nonRepresentableMessage) {
        if (domain.hasZeroElements()) {
            return Layout.of(domain, Stride.zeros(domain));
        }

        long[] inferredStride = inferAffineStride(domain, map, nonRepresentableMessage);
        return Layout.of(domain, Stride.template(domain, inferredStride));
    }

    private static long[] inferAffineStride(
            Shape domain, CoordinateMap map, String nonRepresentableMessage) {
        int rank = domain.flatRank();
        long[] inferred = new long[rank];
        long[] origin = new long[rank];

        long originValue = map.apply(origin);
        if (originValue != 0) {
            throw new IllegalArgumentException(nonRepresentableMessage);
        }

        for (int axis = 0; axis < rank; axis++) {
            if (domain.flatAt(axis) <= 1) {
                inferred[axis] = 0;
                continue;
            }
            long[] basis = origin.clone();
            basis[axis] = 1;
            inferred[axis] = map.apply(basis);
        }

        if (!matchesAffineMapOnDomain(domain, inferred, map)) {
            throw new IllegalArgumentException(nonRepresentableMessage);
        }

        return inferred;
    }

    private static boolean matchesAffineMapOnDomain(
            Shape domain, long[] stride, CoordinateMap map) {
        long[] dims = domain.toArray();
        long[] coord = new long[dims.length];
        while (true) {
            long expected = map.apply(coord);
            long actual = dot(stride, coord);
            if (expected != actual) {
                return false;
            }
            if (!incrementCoord(coord, dims)) {
                return true;
            }
        }
    }

    private static void fillInverseLookup(Layout layout, long[] inverseLinear) {
        Shape domain = layout.shape();
        if (domain.hasZeroElements()) {
            return;
        }
        long[] dims = domain.toArray();
        long[] coord = new long[dims.length];
        while (true) {
            long offset = evaluateOffset(layout, coord);
            if (offset < 0 || offset >= inverseLinear.length) {
                throw new IllegalArgumentException(
                        "inverse requires a zero-based dense codomain [0, size-1]");
            }
            if (inverseLinear[(int) offset] != -1L) {
                throw new IllegalArgumentException("inverse requires a bijective layout");
            }
            long linear = Indexing.coordToLinear(domain, coord);
            inverseLinear[(int) offset] = linear;
            if (!incrementCoord(coord, dims)) {
                break;
            }
        }

        for (long value : inverseLinear) {
            if (value < 0) {
                throw new IllegalArgumentException(
                        "inverse requires a zero-based dense codomain [0, size-1]");
            }
        }
    }

    private static long evaluateOffset(Layout layout, long[] coord) {
        return dot(layout.stride().toArray(), coord);
    }

    private static long dot(long[] stride, long[] coord) {
        if (stride.length != coord.length) {
            throw new IllegalArgumentException(
                    "rank mismatch in dot product: " + stride.length + " vs " + coord.length);
        }
        long value = 0;
        for (int i = 0; i < stride.length; i++) {
            value = Math.addExact(value, Math.multiplyExact(stride[i], coord[i]));
        }
        return value;
    }

    private static boolean incrementCoord(long[] coord, long[] dims) {
        for (int i = dims.length - 1; i >= 0; i--) {
            coord[i]++;
            if (coord[i] < dims[i]) {
                return true;
            }
            coord[i] = 0;
        }
        return false;
    }

    private static SpanInfo spanInfo(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return new SpanInfo(0, -1);
        }
        long min = 0;
        long max = 0;
        long[] dims = layout.shape().toArray();
        long[] strides = layout.stride().toArray();
        for (int i = 0; i < dims.length; i++) {
            long dim = dims[i];
            if (dim <= 1) {
                continue;
            }
            long span = Math.multiplyExact(dim - 1, strides[i]);
            if (span >= 0) {
                max = Math.addExact(max, span);
            } else {
                min = Math.addExact(min, span);
            }
        }
        return new SpanInfo(min, max);
    }

    private static boolean canMerge(long leftStride, long rightDim, long rightStride) {
        return leftStride == Math.multiplyExact(rightDim, rightStride);
    }

    private static List<Mode> normalizedModes(Layout layout) {
        long[] dims = layout.shape().toArray();
        long[] strides = layout.stride().toArray();
        List<Mode> modes = new ArrayList<>(dims.length);
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] <= 1) {
                continue;
            }
            modes.add(new Mode(dims[i], strides[i]));
        }
        return modes;
    }

    private static long ceilDiv(long numerator, long denominator) {
        if (denominator <= 0) {
            throw new IllegalArgumentException("denominator must be > 0");
        }
        long q = numerator / denominator;
        long r = numerator % denominator;
        return r == 0 ? q : Math.addExact(q, 1);
    }

    private static long[] toArray(List<Long> values) {
        long[] result = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }
        return result;
    }

    private record ModePair(long leftDim, long leftStride, long rightDim, long rightStride) {}

    private record Mode(long dim, long stride) {}

    @FunctionalInterface
    private interface CoordinateMap {
        long apply(long[] coord);
    }

    private record SpanInfo(long minOffset, long maxOffset) {}

    public static Layout of(Shape shape, Stride stride) {
        return new LayoutImpl(shape, stride);
    }
}
