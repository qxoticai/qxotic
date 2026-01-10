package com.qxotic.jota.impl;

import com.qxotic.jota.Stride;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class StrideImpl extends NestedTupleImpl<Stride> implements Stride {

    private final int rank;

    StrideImpl(long[] flat, int[] parent) {
        super(flat, parent);
        if (this.parent == null) {
            this.rank = flat.length;
        } else {
            int r = 0;
            for (int p : parent) {
                if (p == -1) {
                    r++;
                }
            }
            this.rank = r;
        }
    }


    @Override
    public int rank() {
        return this.rank;
    }

    @Override
    public Stride modeAt(int _modeIndex) {
        int modeIndex = com.qxotic.jota.Util.wrapAround(_modeIndex, rank());

        if (isFlat()) {
            return StrideImpl.of(new long[]{flat[modeIndex]});
        }

        // Find the modeIndex-th root in the parent array
        int rootCount = 0;
        int rootIndex = -1;
        for (int i = 0; i < parent.length; i++) {
            if (parent[i] == -1) {
                if (rootCount == modeIndex) {
                    rootIndex = i;
                    break;
                }
                rootCount++;
            }
        }

        if (rootIndex == -1) {
            throw new IndexOutOfBoundsException("Mode index " + modeIndex + " out of bounds for rank " + rank());
        }

        // Collect all dimensions that belong to this mode (root and its descendants)
        ArrayList<Long> modeStrides = new ArrayList<>();
        ArrayList<Integer> modeParents = new ArrayList<>();

        // Add root stride
        modeStrides.add(flat[rootIndex]);

        // Scan forward for descendants (until we hit the next top-level root)
        for (int i = rootIndex + 1; i < parent.length && parent[i] != -1; i++) {
            modeStrides.add(flat[i]);
        }

        // Build the new parent array
        // Direct children of the root become new roots at this level
        // Descendants of direct children are remapped accordingly
        for (int i = 0; i < modeStrides.size(); i++) {
            int absIndex = rootIndex + i;
            if (i == 0) {
                // First element is the root
                modeParents.add(-1);
            } else {
                int originalParent = parent[absIndex];
                if (originalParent == rootIndex) {
                    // Direct child of root -> becomes a root at this level
                    modeParents.add(-1);
                } else {
                    // Child of some other node -> remap the parent index
                    int remappedParent = originalParent - rootIndex;
                    modeParents.add(remappedParent);
                }
            }
        }

        // Convert to arrays
        long[] modeStridesArray = new long[modeStrides.size()];
        int[] modeParentsArray = new int[modeParents.size()];
        for (int i = 0; i < modeStrides.size(); i++) {
            modeStridesArray[i] = modeStrides.get(i);
            modeParentsArray[i] = modeParents.get(i);
        }

        // Check if this mode has nested structure (multiple roots)
        int nestedRootCount = 0;
        for (int p : modeParentsArray) {
            if (p == -1) nestedRootCount++;
        }

        // If this mode is just a simple list/tree with one root, return it as flat
        // Otherwise, preserve the nesting structure
        if (nestedRootCount <= 1) {
            return StrideImpl.of(modeStridesArray);
        } else {
            return new StrideImpl(modeStridesArray, modeParentsArray);
        }
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

    static Stride of(long[] strides, int[] parent) {
        if (strides.length == 0) {
            return scalar();
        }
        long[] stridesCopy = strides.clone();
        int[] parentCopy = null;
        if (parent != null) {
            parentCopy = parent.clone();
        }
        return new StrideImpl(stridesCopy, parentCopy);
    }

    static Stride singleton(long stride) {
        return new StrideImpl(new long[]{stride}, null);
    }

    static Stride nested(Object... elements) {
        // Single argument case - unwrap if it's a Stride
        if (elements.length == 1) {
            if (elements[0] instanceof Stride stride) {
                return stride;
            }
            if (elements[0] instanceof Number num) {
                return StrideImpl.of(new long[]{num.longValue()});
            }
            throw new IllegalArgumentException("Arguments must be Numbers or Strides");
        }

        // Multiple arguments - build composite structure
        ArrayList<Long> flatStrides = new ArrayList<>();
        ArrayList<Integer> parents = new ArrayList<>();

        Object[] allArgs = elements;

        for (Object arg : allArgs) {
            if (arg instanceof Number num) {
                // Add as single stride at current level
                flatStrides.add(num.longValue());
                parents.add(-1);
            } else if (arg instanceof Stride stride) {
                if (stride.flatRank() == 1) {
                    // Single-element stride - unwrap to single value
                    flatStrides.add(stride.flatAt(0));
                    parents.add(-1);
                } else {
                    // Multi-element stride - add as nested group
                    int groupRoot = flatStrides.size();
                    flatStrides.add(stride.flatAt(0));
                    parents.add(-1); // First element is root at this level

                    // Add remaining elements as children
                    for (int i = 1; i < stride.flatRank(); i++) {
                        flatStrides.add(stride.flatAt(i));

                        // Determine parent
                        if (stride instanceof StrideImpl impl && impl.parent != null) {
                            int originalParent = impl.parent[i];
                            if (originalParent == -1) {
                                // Another root in the nested stride -> child of group root
                                parents.add(groupRoot);
                            } else {
                                // Child of some element -> remap relative to group root
                                parents.add(groupRoot + originalParent);
                            }
                        } else {
                            // Flat stride - all are children of the first element
                            parents.add(groupRoot);
                        }
                    }
                }
            } else {
                throw new IllegalArgumentException("Arguments must be Numbers or Strides");
            }
        }

        // Convert to arrays
        long[] strides = new long[flatStrides.size()];
        int[] parentArray = new int[parents.size()];
        for (int i = 0; i < flatStrides.size(); i++) {
            strides[i] = flatStrides.get(i);
            parentArray[i] = parents.get(i);
        }

        return new StrideImpl(strides, parentArray);
    }

    static Stride scalar() {
        return SCALAR;
    }

    @Override
    public Stride replace(int _modeIndex, Stride newMode) {
        int modeIndex = com.qxotic.jota.Util.wrapAround(_modeIndex, rank());

        if (isFlat()) {
            List<Long> newFlat = new ArrayList<>();
            List<Integer> newParent = new ArrayList<>();

            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    if (newMode.flatRank() == 1) {
                        newFlat.add(newMode.flatAt(0));
                        newParent.add(-1);
                    } else {
                        int groupRoot = newFlat.size();
                        for (int j = 0; j < newMode.flatRank(); j++) {
                            newFlat.add(newMode.flatAt(j));
                            if (j == 0) {
                                newParent.add(-1);
                            } else if (newMode instanceof StrideImpl impl && impl.parent != null) {
                                int originalParent = impl.parent[j];
                                newParent.add(originalParent == -1 ? groupRoot : groupRoot + originalParent);
                            } else {
                                newParent.add(groupRoot);
                            }
                        }
                    }
                } else {
                    newFlat.add(flat[i]);
                    newParent.add(-1);
                }
            }

            return buildStride(newFlat, newParent);
        }

        ModeRange range = findModeRange(modeIndex);
        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        int insertPoint = newFlat.size();
        if (newMode.flatRank() == 1) {
            newFlat.add(newMode.flatAt(0));
            newParent.add(-1);
        } else {
            for (int j = 0; j < newMode.flatRank(); j++) {
                newFlat.add(newMode.flatAt(j));
                if (j == 0) {
                    newParent.add(-1);
                } else if (newMode instanceof StrideImpl impl && impl.parent != null) {
                    int originalParent = impl.parent[j];
                    newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                } else {
                    newParent.add(insertPoint);
                }
            }
        }

        int offset = newFlat.size() - range.end;
        for (int i = range.end; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildStride(newFlat, newParent);
    }

    @Override
    public Stride insert(int _modeIndex, Stride mode) {
        int modeIndex = com.qxotic.jota.Util.wrapAround(_modeIndex, rank() + 1);

        if (isScalar()) {
            return mode;
        }

        if (isFlat()) {
            List<Long> newFlat = new ArrayList<>();
            List<Integer> newParent = new ArrayList<>();

            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    int insertPoint = newFlat.size();
                    if (mode.flatRank() == 1) {
                        newFlat.add(mode.flatAt(0));
                        newParent.add(-1);
                    } else {
                        for (int j = 0; j < mode.flatRank(); j++) {
                            newFlat.add(mode.flatAt(j));
                            if (j == 0) {
                                newParent.add(-1);
                            } else if (mode instanceof StrideImpl impl && impl.parent != null) {
                                int originalParent = impl.parent[j];
                                newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                            } else {
                                newParent.add(insertPoint);
                            }
                        }
                    }
                }
                newFlat.add(flat[i]);
                newParent.add(-1);
            }

            if (modeIndex >= flat.length) {
                int insertPoint = newFlat.size();
                if (mode.flatRank() == 1) {
                    newFlat.add(mode.flatAt(0));
                    newParent.add(-1);
                } else {
                    for (int j = 0; j < mode.flatRank(); j++) {
                        newFlat.add(mode.flatAt(j));
                        if (j == 0) {
                            newParent.add(-1);
                        } else if (mode instanceof StrideImpl impl && impl.parent != null) {
                            int originalParent = impl.parent[j];
                            newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                        } else {
                            newParent.add(insertPoint);
                        }
                    }
                }
            }

            return buildStride(newFlat, newParent);
        }

        ModeRange range = modeIndex < rank() ? findModeRange(modeIndex) : new ModeRange(flat.length, flat.length);
        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        int insertPoint = newFlat.size();
        if (mode.flatRank() == 1) {
            newFlat.add(mode.flatAt(0));
            newParent.add(-1);
        } else {
            for (int j = 0; j < mode.flatRank(); j++) {
                newFlat.add(mode.flatAt(j));
                if (j == 0) {
                    newParent.add(-1);
                } else if (mode instanceof StrideImpl impl && impl.parent != null) {
                    int originalParent = impl.parent[j];
                    newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                } else {
                    newParent.add(insertPoint);
                }
            }
        }

        int offset = newFlat.size() - range.start;
        for (int i = range.start; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildStride(newFlat, newParent);
    }

    @Override
    public Stride remove(int _modeIndex) {
        int modeIndex = com.qxotic.jota.Util.wrapAround(_modeIndex, rank());

        if (rank() == 1) {
            return scalar();
        }

        if (isFlat()) {
            List<Long> newFlat = new ArrayList<>();
            for (int i = 0; i < flat.length; i++) {
                if (i != modeIndex) {
                    newFlat.add(flat[i]);
                }
            }
            return StrideImpl.of(newFlat.stream().mapToLong(Long::longValue).toArray());
        }

        ModeRange range = findModeRange(modeIndex);
        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        int offset = range.start - range.end;
        for (int i = range.end; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildStride(newFlat, newParent);
    }

    @Override
    public Stride permute(int... _modeIndices) {
        if (_modeIndices.length != rank()) {
            throw new IllegalArgumentException("Permutation must have same length as rank: " + rank() + " vs " + _modeIndices.length);
        }

        int[] axes = new int[_modeIndices.length];
        for (int i = 0; i < _modeIndices.length; i++) {
            axes[i] = com.qxotic.jota.Util.wrapAround(_modeIndices[i], rank());
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
        List<Integer> newParent = new ArrayList<>();

        for (int newPos = 0; newPos < axes.length; newPos++) {
            int oldPos = axes[newPos];
            ModeRange range = ranges[oldPos];
            int modeStart = newFlat.size();

            for (int i = range.start; i < range.end; i++) {
                newFlat.add(flat[i]);
                if (i == range.start) {
                    newParent.add(-1);
                } else {
                    int p = parent[i];
                    if (p == -1) {
                        newParent.add(modeStart);
                    } else {
                        newParent.add(modeStart + (p - range.start));
                    }
                }
            }
        }

        return buildStride(newFlat, newParent);
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
        // Parent array remains the same
        return new StrideImpl(scaledFlat, parent);
    }

    private ModeRange findModeRange(int modeIndex) {
        int rootCount = 0;
        int start = -1;
        for (int i = 0; i < parent.length; i++) {
            if (parent[i] == -1) {
                if (rootCount == modeIndex) {
                    start = i;
                    break;
                }
                rootCount++;
            }
        }

        if (start == -1) {
            throw new IndexOutOfBoundsException("Mode index out of bounds: " + modeIndex);
        }

        int end = start + 1;
        while (end < parent.length && parent[end] != -1) {
            end++;
        }

        return new ModeRange(start, end);
    }

    private static class ModeRange {
        final int start;
        final int end;

        ModeRange(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    private Stride buildStride(List<Long> flatList, List<Integer> parentList) {
        if (flatList.isEmpty()) {
            return scalar();
        }

        long[] flatArray = new long[flatList.size()];
        int[] parentArray = new int[parentList.size()];
        for (int i = 0; i < flatList.size(); i++) {
            flatArray[i] = flatList.get(i);
            parentArray[i] = parentList.get(i);
        }

        boolean allRoots = true;
        for (int p : parentArray) {
            if (p != -1) {
                allRoots = false;
                break;
            }
        }

        if (allRoots) {
            return StrideImpl.of(flatArray);
        } else {
            return new StrideImpl(flatArray, parentArray);
        }
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof StrideImpl that)
                && Arrays.equals(this.flat, that.flat)
                && Arrays.equals(this.parent, that.parent);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(flat), Arrays.hashCode(parent));
    }
}
