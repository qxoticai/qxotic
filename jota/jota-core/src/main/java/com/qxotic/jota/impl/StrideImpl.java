package com.qxotic.jota.impl;

import com.qxotic.jota.Stride;
import com.qxotic.jota.Util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class StrideImpl extends NestedTupleImpl<Stride> implements Stride {

    private final int rank;

    StrideImpl(long[] flat, int[] nest) {
        super(flat, nest);
        if (this.nest == null) {
            this.rank = flat.length;
        } else {
            this.rank = countTopLevelModes(this.nest);
        }
    }

    @Override
    public int rank() {
        return this.rank;
    }

    @Override
    public Stride modeAt(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        if (isFlat()) {
            return StrideImpl.of(flat[modeIndex]);
        }

        ModeRange range = findModeRange(modeIndex);
        long[] modeStridesArray = Arrays.copyOfRange(flat, range.start, range.end);
        int[] modeNestArray =
                nest == null ? null : Arrays.copyOfRange(nest, range.start, range.end);

        if (modeNestArray != null && modeNestArray.length > 0 && modeNestArray[0] > 0) {
            modeNestArray[0] -= 1;
            int last = modeNestArray.length - 1;
            modeNestArray[last] += 1;
        }

        if (isFlatNest(modeNestArray)) {
            return StrideImpl.of(modeStridesArray);
        }
        return new StrideImpl(modeStridesArray, modeNestArray);
    }

    @Override
    public Stride flatten() {
        if (isFlat()) {
            return this;
        } else {
            return StrideImpl.of(flat);
        }
    }

    private static final Stride SCALAR = new StrideImpl(EMPTY, null);

    static Stride flat(long... strides) {
        if (strides.length == 0) {
            return scalar();
        }
        return new StrideImpl(strides.clone(), null);
    }

    static Stride of(long... strides) {
        return flat(strides);
    }

    static Stride of(long[] strides, int[] nest) {
        if (strides.length == 0) {
            return scalar();
        }
        long[] stridesCopy = strides.clone();
        int[] nestCopy = nest == null ? null : nest.clone();
        if (isFlatNest(nestCopy)) {
            nestCopy = null;
        }
        return new StrideImpl(stridesCopy, nestCopy);
    }

    static Stride singleton(long stride) {
        return new StrideImpl(new long[] {stride}, null);
    }

    static Stride nested(Object... elements) {
        // Single argument case - unwrap if it's a Stride
        if (elements.length == 1) {
            if (elements[0] instanceof Stride stride) {
                return stride;
            }
            if (elements[0] instanceof Number num) {
                return StrideImpl.of(num.longValue());
            }
            throw new IllegalArgumentException("Arguments must be Numbers or Strides");
        }

        // Multiple arguments - build composite structure
        ArrayList<Long> flatStrides = new ArrayList<>();
        ArrayList<Integer> nests = new ArrayList<>();

        for (Object arg : elements) {
            if (arg instanceof Number num) {
                flatStrides.add(num.longValue());
                nests.add(0);
            } else if (arg instanceof Stride stride) {
                appendMode(flatStrides, nests, stride);
            } else {
                throw new IllegalArgumentException("Arguments must be Numbers or Strides");
            }
        }

        return buildStride(flatStrides, nests);
    }

    static Stride scalar() {
        return SCALAR;
    }

    @Override
    public Stride replace(int _modeIndex, Stride newMode) {
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
            return buildStride(newFlat, newNest);
        }

        ModeRange range = findModeRange(modeIndex);
        appendRange(newFlat, newNest, 0, range.start);
        appendMode(newFlat, newNest, newMode);
        appendRange(newFlat, newNest, range.end, flat.length);
        return buildStride(newFlat, newNest);
    }

    @Override
    public Stride insert(int modeIndex_, Stride mode) {
        int modeIndex = Util.wrapAround(modeIndex_, rank() + 1);

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
            return buildStride(newFlat, newNest);
        }

        int insertIndex = modeIndex < rank() ? findModeRange(modeIndex).start : flat.length;
        appendRange(newFlat, newNest, 0, insertIndex);
        appendMode(newFlat, newNest, mode);
        appendRange(newFlat, newNest, insertIndex, flat.length);
        return buildStride(newFlat, newNest);
    }

    @Override
    public Stride remove(int _modeIndex) {
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
            return buildStride(newFlat, newNest);
        }

        ModeRange range = findModeRange(modeIndex);
        List<Long> newFlat = new ArrayList<>();
        List<Integer> newNest = new ArrayList<>();
        appendRange(newFlat, newNest, 0, range.start);
        appendRange(newFlat, newNest, range.end, flat.length);
        return buildStride(newFlat, newNest);
    }

    @Override
    public Stride permute(int... _modeIndices) {
        if (_modeIndices.length != rank()) {
            throw new IllegalArgumentException(
                    "Permutation must have same length as rank: "
                            + rank()
                            + " vs "
                            + _modeIndices.length);
        }

        int[] axes = new int[_modeIndices.length];
        for (int i = 0; i < _modeIndices.length; i++) {
            axes[i] = Util.wrapAround(_modeIndices[i], rank());
        }

        boolean[] seen = new boolean[rank()];
        for (int axis : axes) {
            if (seen[axis]) {
                throw new IllegalArgumentException("Duplicate axis in permutation: " + axis);
            }
            seen[axis] = true;
        }

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
            long[] newFlat = new long[flat.length];
            for (int i = 0; i < axes.length; i++) {
                newFlat[i] = flat[axes[i]];
            }
            return StrideImpl.of(newFlat);
        }

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

        return buildStride(newFlat, newNest);
    }

    @Override
    public Stride scale(long factor) {
        if (factor == 1) {
            return this;
        }
        long[] scaledFlat = new long[flat.length];
        for (int i = 0; i < flat.length; i++) {
            scaledFlat[i] = flat[i] * factor;
        }
        // Nest array remains the same
        return new StrideImpl(scaledFlat, nest);
    }

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

    private record ModeRange(int start, int end) {}

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

    private static int[] extractNest(Stride mode) {
        if (mode instanceof StrideImpl impl) {
            return impl.nest;
        }
        if (mode.isFlat()) {
            return null;
        }
        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }

    private static void appendMode(List<Long> flatList, List<Integer> nestList, Stride mode) {
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

    private static Stride buildStride(List<Long> flatList, List<Integer> nestList) {
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
            return StrideImpl.of(flatArray);
        }
        return new StrideImpl(flatArray, nestArray);
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof StrideImpl that)
                && Arrays.equals(this.flat, that.flat)
                && Arrays.equals(this.nest, that.nest);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(flat), Arrays.hashCode(nest));
    }
}
