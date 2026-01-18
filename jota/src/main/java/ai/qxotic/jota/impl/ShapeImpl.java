package ai.qxotic.jota.impl;

import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class ShapeImpl extends NestedTupleImpl<Shape> implements Shape {

    private static final Shape SCALAR = new ShapeImpl(EMPTY, null);

    // Derived properties, pre-computed for performance.
    private final long[] modeSize;
    private final long totalSize;

    ShapeImpl(long[] flat, int[] nest) {
        super(checkNoNegatives(flat), nest);
        this.totalSize = Arrays.stream(this.flat).reduce(1L, Math::multiplyExact);
        if (this.nest == null) {
            this.modeSize = this.flat;
            return;
        }

        int rank = countTopLevelModes(this.nest);
        this.modeSize = new long[rank];
        int depth = 0;
        int modeIdx = -1;
        long product = 1L;
        for (int i = 0; i < flat.length; i++) {
            if (depth == 0) {
                modeIdx++;
                product = 1L;
            }
            int open = Math.max(nest[i], 0);
            int close = Math.max(-nest[i], 0);
            depth += open;
            product *= flat[i];
            depth -= close;
            if (depth == 0) {
                modeSize[modeIdx] = product;
            }
        }
    }

    private static long[] checkNoNegatives(long[] flat) {
        for (long dim : flat) {
            if (dim < 0) {
                throw new IllegalArgumentException("negative dimension");
            }
        }
        return flat;
    }

    @Override
    public int rank() {
        return modeSize.length;
    }

    @Override
    public Shape modeAt(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());
        if (isFlat()) {
            return ShapeImpl.flat(flatAt(modeIndex));
        }
        ModeRange range = findModeRange(modeIndex);
        long[] modeDimsArray = Arrays.copyOfRange(flat, range.start, range.end);
        int[] modeNestArray = Arrays.copyOfRange(nest, range.start, range.end);
        if (modeNestArray[0] > 0) {
            modeNestArray[0] -= 1;
            modeNestArray[modeNestArray.length - 1] += 1;
        }
        if (isFlatNest(modeNestArray)) {
            return ShapeImpl.of(modeDimsArray);
        }
        return new ShapeImpl(modeDimsArray, modeNestArray);
    }

    @Override
    public long size(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());
        return modeSize[modeIndex];
    }

    @Override
    public long size() {
        return totalSize;
    }

    @Override
    public Shape flattenModes() {
        return flat(modeSize);
    }

    @Override
    public Shape flatten() {
        if (isFlat()) {
            return this;
        } else {
            return ShapeImpl.of(flat);
        }
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof ShapeImpl that)
                && Arrays.equals(this.flat, that.flat)
                && Arrays.equals(this.nest, that.nest);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(flat), Arrays.hashCode(nest));
    }

    static Shape scalar() {
        return SCALAR;
    }

    static Shape nested(Object... elements) {
        // Single argument case - unwrap if it's a Shape
        if (elements.length == 1) {
            if (elements[0] instanceof Shape shape) {
                return shape;
            }
            if (elements[0] instanceof Number num) {
                return ShapeImpl.of(num.longValue());
            }
            throw new IllegalArgumentException("Arguments must be Numbers or Shapes");
        }

        // Multiple arguments - build composite structure
        List<Long> flatDims = new ArrayList<>();
        List<Integer> nests = new ArrayList<>();

        for (Object arg : elements) {
            if (arg instanceof Number num) {
                flatDims.add(num.longValue());
                nests.add(0);
            } else if (arg instanceof Shape shape) {
                appendMode(flatDims, nests, shape);
            } else {
                throw new IllegalArgumentException("Arguments must be Numbers or Shapes");
            }
        }

        return buildShape(flatDims, nests);
    }

    static Shape flat(long... dims) {
        if (dims.length == 0) {
            return scalar();
        }
        return new ShapeImpl(dims.clone(), null);
    }

    static Shape of(long... dims) {
        return flat(dims);
    }

    static Shape of(long[] dims, int[] nest) {
        if (dims.length == 0) {
            return scalar();
        }
        int[] nestCopy = isFlatNest(nest) ? null : nest.clone();
        return new ShapeImpl(dims.clone(), nestCopy);
    }

    @Override
    public Shape replace(int _modeIndex, Shape newMode) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newNest = new ArrayList<>();

        if (isFlat()) {
            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    appendMode(newFlat, newNest, newMode);
                } else {
                    newFlat.add(flat[i]);
                    newNest.add(0);
                }
            }
            return buildShape(newFlat, newNest);
        }

        ModeRange range = findModeRange(modeIndex);
        appendRange(newFlat, newNest, 0, range.start);
        appendMode(newFlat, newNest, newMode);
        appendRange(newFlat, newNest, range.end, flat.length);
        return buildShape(newFlat, newNest);
    }

    @Override
    public Shape insert(int _modeIndex, Shape mode) {
        int modeIndex = Util.wrapAround(_modeIndex, rank() + 1); // Allow inserting at rank()

        if (isScalar()) {
            return mode;
        }

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newNest = new ArrayList<>();

        if (isFlat()) {
            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    appendMode(newFlat, newNest, mode);
                }
                newFlat.add(flat[i]);
                newNest.add(0);
            }
            if (modeIndex >= flat.length) {
                appendMode(newFlat, newNest, mode);
            }
            return buildShape(newFlat, newNest);
        }

        int insertIndex = modeIndex < rank() ? findModeRange(modeIndex).start : flat.length;
        appendRange(newFlat, newNest, 0, insertIndex);
        appendMode(newFlat, newNest, mode);
        appendRange(newFlat, newNest, insertIndex, flat.length);
        return buildShape(newFlat, newNest);
    }

    @Override
    public Shape remove(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        if (rank() == 1) {
            return scalar();
        }

        if (isFlat()) {
            List<Long> newFlat = new ArrayList<>();
            List<Integer> newNest = new ArrayList<>();
            for (int i = 0; i < flat.length; i++) {
                if (i != modeIndex) {
                    newFlat.add(flat[i]);
                    newNest.add(0);
                }
            }
            return buildShape(newFlat, newNest);
        }

        ModeRange range = findModeRange(modeIndex);
        List<Long> newFlat = new ArrayList<>();
        List<Integer> newNest = new ArrayList<>();
        appendRange(newFlat, newNest, 0, range.start);
        appendRange(newFlat, newNest, range.end, flat.length);
        return buildShape(newFlat, newNest);
    }

    @Override
    public Shape permute(int... _modeIndices) {
        if (_modeIndices.length != rank()) {
            throw new IllegalArgumentException(
                    "Permutation must have same length as rank: "
                            + rank()
                            + " vs "
                            + _modeIndices.length);
        }

        // Normalize negative indices
        int[] axes = new int[_modeIndices.length];
        for (int i = 0; i < _modeIndices.length; i++) {
            axes[i] = Util.wrapAround(_modeIndices[i], rank());
        }

        // Validate: each index must appear exactly once
        boolean[] seen = new boolean[rank()];
        for (int axis : axes) {
            if (seen[axis]) {
                throw new IllegalArgumentException("Duplicate axis in permutation: " + axis);
            }
            seen[axis] = true;
        }

        // Identity permutation - return this
        boolean isIdentity = true;
        for (int i = 0; i < axes.length; i++) {
            if (axes[i] != i) {
                isIdentity = false;
                break;
            }
        }
        if (isIdentity) {
            return this;
        }

        if (isFlat()) {
            // Simple case: reorder dimensions
            long[] newFlat = new long[flat.length];
            for (int i = 0; i < axes.length; i++) {
                newFlat[i] = flat[axes[i]];
            }
            return ShapeImpl.of(newFlat);
        }

        // For nested shapes: extract modes and reorder them
        ModeRange[] ranges = new ModeRange[rank()];
        for (int i = 0; i < rank(); i++) {
            ranges[i] = findModeRange(i);
        }

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newNest = new ArrayList<>();

        for (int newPos = 0; newPos < axes.length; newPos++) {
            int oldPos = axes[newPos];
            ModeRange range = ranges[oldPos];
            appendRange(newFlat, newNest, range.start, range.end);
        }

        return buildShape(newFlat, newNest);
    }

    // Helper to find the range of flat indices for a mode
    private ModeRange findModeRange(int modeIndex) {
        int depth = 0;
        int currentMode = -1;
        int start = -1;
        for (int i = 0; i < nest.length; i++) {
            if (depth == 0) {
                currentMode++;
                if (currentMode == modeIndex) {
                    start = i;
                }
            }
            depth += Math.max(nest[i], 0);
            depth -= Math.max(-nest[i], 0);
            if (currentMode == modeIndex && depth == 0) {
                return new ModeRange(start, i + 1);
            }
        }

        throw new IndexOutOfBoundsException("Mode index out of bounds: " + modeIndex);
    }

    private static class ModeRange {
        final int start;
        final int end;

        ModeRange(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    private static boolean isFlatNest(int[] nest) {
        if (nest == null) {
            return true;
        }
        for (int value : nest) {
            if (value != 0) {
                return false;
            }
        }
        return true;
    }

    private static boolean isFlatNest(List<Integer> nestList) {
        for (int value : nestList) {
            if (value != 0) {
                return false;
            }
        }
        return true;
    }

    private static int countTopLevelModes(int[] nest) {
        int depth = 0;
        int count = 0;
        for (int value : nest) {
            if (depth == 0) {
                count++;
            }
            depth += Math.max(value, 0);
            depth -= Math.max(-value, 0);
        }
        return count;
    }

    private static int[] extractNest(Shape mode) {
        if (mode instanceof ShapeImpl impl) {
            return impl.nest;
        }
        if (mode.isFlat()) {
            return null;
        }
        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }

    private static void appendMode(List<Long> flatList, List<Integer> nestList, Shape mode) {
        int[] modeNest = extractNest(mode);
        if (mode.flatRank() == 1 && isFlatNest(modeNest)) {
            flatList.add(mode.flatAt(0));
            nestList.add(0);
            return;
        }

        for (int i = 0; i < mode.flatRank(); i++) {
            flatList.add(mode.flatAt(i));
            int nestValue = modeNest == null ? 0 : modeNest[i];
            if (i == 0) {
                nestValue += 1;
            }
            if (i == mode.flatRank() - 1) {
                nestValue -= 1;
            }
            nestList.add(nestValue);
        }
    }

    private void appendRange(List<Long> flatList, List<Integer> nestList, int start, int end) {
        for (int i = start; i < end; i++) {
            flatList.add(flat[i]);
            nestList.add(nest == null ? 0 : nest[i]);
        }
    }

    // Helper to build shape from lists
    private static Shape buildShape(List<Long> flatList, List<Integer> nestList) {
        if (flatList.isEmpty()) {
            return scalar();
        }

        long[] flatArray = new long[flatList.size()];
        int[] nestArray = new int[nestList.size()];
        for (int i = 0; i < flatList.size(); i++) {
            flatArray[i] = flatList.get(i);
            nestArray[i] = nestList.get(i);
        }

        if (isFlatNest(nestArray)) {
            return ShapeImpl.flat(flatArray);
        }
        return new ShapeImpl(flatArray, nestArray);
    }
}
