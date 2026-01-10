package com.qxotic.jota.impl;

import com.qxotic.jota.Shape;
import com.qxotic.jota.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class ShapeImpl extends NestedTupleImpl<Shape> implements Shape {

    private static final Shape SCALAR = new ShapeImpl(EMPTY, null);

    // Derived, pre-computed for performance.
    private final long[] modeSize;
    private final long totalSize;

    ShapeImpl(long[] flat, int[] parent) {
        super(flat, parent);

        if (Arrays.stream(flat).anyMatch(dim -> dim < 0)) {
            throw new IllegalArgumentException("negative dimension");
        }

        this.totalSize = Arrays.stream(this.flat).reduce(1L, Math::multiplyExact);
        if (this.parent == null) {
            this.modeSize = this.flat;
            return;
        }

        // Compute rank (count roots)
        int rank = 0;
        for (int p : parent) {
            if (p == -1) rank++;
        }
        this.modeSize = new long[rank];
        int modeIdx = 0;
        for (int i = 0; i < flat.length; i++) {
            if (parent[i] == -1) {
                // Found a root - compute product of this mode's subtree
                long product = flat[i];
                // Scan forward for children (until next root or end)
                for (int j = i + 1; j < flat.length && parent[j] != -1; j++) {
                    product *= flat[j];
                }
                modeSize[modeIdx++] = product;
            }
        }
    }

    @Override
    public int rank() {
        return modeSize.length;
    }

    @Override
    public Shape modeAt(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        if (isFlat()) {
            return ShapeImpl.of(new long[]{flatAt(modeIndex)});
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
        ArrayList<Long> modeDims = new ArrayList<>();
        ArrayList<Integer> modeParents = new ArrayList<>();

        // Add root dimension
        modeDims.add(flat[rootIndex]);

        // Scan forward for descendants (until we hit the next top-level root)
        for (int i = rootIndex + 1; i < parent.length && parent[i] != -1; i++) {
            modeDims.add(flat[i]);
        }

        // Build the new parent array
        // Direct children of the root become new roots at this level
        // Descendants of direct children are remapped accordingly
        for (int i = 0; i < modeDims.size(); i++) {
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
        long[] modeDimsArray = new long[modeDims.size()];
        int[] modeParentsArray = new int[modeParents.size()];
        for (int i = 0; i < modeDims.size(); i++) {
            modeDimsArray[i] = modeDims.get(i);
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
            return ShapeImpl.of(modeDimsArray);
        } else {
            return new ShapeImpl(modeDimsArray, modeParentsArray);
        }
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
                && Arrays.equals(this.parent, that.parent);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(flat), Arrays.hashCode(parent));
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
        List<Integer> parents = new ArrayList<>();

        Object[] allArgs = elements;

        for (Object arg : allArgs) {
            if (arg instanceof Number num) {
                // Add as single dimension at current level
                flatDims.add(num.longValue());
                parents.add(-1);
            } else if (arg instanceof Shape shape) {
                if (shape.flatRank() == 1) {
                    // Single-element shape - unwrap to single dimension
                    flatDims.add(shape.flatAt(0));
                    parents.add(-1);
                } else {
                    // Multi-element shape - add as nested group
                    int groupRoot = flatDims.size();
                    flatDims.add(shape.flatAt(0));
                    parents.add(-1); // First element is root at this level

                    // Add remaining elements as children
                    for (int i = 1; i < shape.flatRank(); i++) {
                        flatDims.add(shape.flatAt(i));

                        // Determine parent
                        if (shape instanceof ShapeImpl impl && impl.parent != null) {
                            int originalParent = impl.parent[i];
                            if (originalParent == -1) {
                                // Another root in the nested shape -> child of group root
                                parents.add(groupRoot);
                            } else {
                                // Child of some element -> remap relative to group root
                                parents.add(groupRoot + originalParent);
                            }
                        } else {
                            // Flat shape - all are children of the first element
                            parents.add(groupRoot);
                        }
                    }
                }
            } else {
                throw new IllegalArgumentException("Arguments must be Numbers or Shapes");
            }
        }

        // Convert to arrays
        long[] dims = new long[flatDims.size()];
        int[] parentArray = new int[parents.size()];
        for (int i = 0; i < flatDims.size(); i++) {
            dims[i] = flatDims.get(i);
            parentArray[i] = parents.get(i);
        }

        return new ShapeImpl(dims, parentArray);
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

    static Shape of(long[] dims, int[] parent) {
        if (dims.length == 0) {
            return scalar();
        }
        return new ShapeImpl(dims.clone(), parent);
    }

    @Override
    public Shape replace(int _modeIndex, Shape newMode) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        if (isFlat()) {
            // For flat shapes, replace a single dimension with newMode's dimensions
            List<Long> newFlat = new ArrayList<>();
            List<Integer> newParent = new ArrayList<>();

            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    // Insert the new mode here
                    if (newMode.flatRank() == 1) {
                        // Single dimension - keep flat
                        newFlat.add(newMode.flatAt(0));
                        newParent.add(-1);
                    } else {
                        // Multiple dimensions - add as nested group
                        int groupRoot = newFlat.size();
                        for (int j = 0; j < newMode.flatRank(); j++) {
                            newFlat.add(newMode.flatAt(j));
                            if (j == 0) {
                                newParent.add(-1);
                            } else if (newMode instanceof ShapeImpl impl && impl.parent != null) {
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

            return buildShape(newFlat, newParent);
        }

        // For nested shapes, find the mode and replace it
        ModeRange range = findModeRange(modeIndex);

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        // Add dimensions before the mode
        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        // Insert the new mode
        int insertPoint = newFlat.size();
        if (newMode.flatRank() == 1) {
            newFlat.add(newMode.flatAt(0));
            newParent.add(-1);
        } else {
            for (int j = 0; j < newMode.flatRank(); j++) {
                newFlat.add(newMode.flatAt(j));
                if (j == 0) {
                    newParent.add(-1);
                } else if (newMode instanceof ShapeImpl impl && impl.parent != null) {
                    int originalParent = impl.parent[j];
                    newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                } else {
                    newParent.add(insertPoint);
                }
            }
        }

        // Add dimensions after the mode, adjusting parent indices
        int offset = newFlat.size() - range.end;
        for (int i = range.end; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildShape(newFlat, newParent);
    }

    @Override
    public Shape insert(int _modeIndex, Shape mode) {
        int modeIndex = Util.wrapAround(_modeIndex, rank() + 1); // Allow inserting at rank()

        if (isScalar()) {
            return mode;
        }

        if (isFlat()) {
            List<Long> newFlat = new ArrayList<>();
            List<Integer> newParent = new ArrayList<>();

            for (int i = 0; i < flat.length; i++) {
                if (i == modeIndex) {
                    // Insert the new mode here
                    int insertPoint = newFlat.size();
                    if (mode.flatRank() == 1) {
                        newFlat.add(mode.flatAt(0));
                        newParent.add(-1);
                    } else {
                        for (int j = 0; j < mode.flatRank(); j++) {
                            newFlat.add(mode.flatAt(j));
                            if (j == 0) {
                                newParent.add(-1);
                            } else if (mode instanceof ShapeImpl impl && impl.parent != null) {
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

            // Handle insertion at the end
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
                        } else if (mode instanceof ShapeImpl impl && impl.parent != null) {
                            int originalParent = impl.parent[j];
                            newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                        } else {
                            newParent.add(insertPoint);
                        }
                    }
                }
            }

            return buildShape(newFlat, newParent);
        }

        // For nested shapes
        ModeRange range = modeIndex < rank() ? findModeRange(modeIndex) : new ModeRange(flat.length, flat.length);

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        // Add dimensions before insertion point
        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        // Insert the new mode
        int insertPoint = newFlat.size();
        if (mode.flatRank() == 1) {
            newFlat.add(mode.flatAt(0));
            newParent.add(-1);
        } else {
            for (int j = 0; j < mode.flatRank(); j++) {
                newFlat.add(mode.flatAt(j));
                if (j == 0) {
                    newParent.add(-1);
                } else if (mode instanceof ShapeImpl impl && impl.parent != null) {
                    int originalParent = impl.parent[j];
                    newParent.add(originalParent == -1 ? insertPoint : insertPoint + originalParent);
                } else {
                    newParent.add(insertPoint);
                }
            }
        }

        // Add remaining dimensions, adjusting parent indices
        int offset = newFlat.size() - range.start;
        for (int i = range.start; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildShape(newFlat, newParent);
    }

    @Override
    public Shape remove(int _modeIndex) {
        int modeIndex = Util.wrapAround(_modeIndex, rank());

        if (rank() == 1) {
            return scalar();
        }

        if (isFlat()) {
            // Remove a single dimension
            List<Long> newFlat = new ArrayList<>();
            for (int i = 0; i < flat.length; i++) {
                if (i != modeIndex) {
                    newFlat.add(flat[i]);
                }
            }
            return ShapeImpl.of(newFlat.stream().mapToLong(Long::longValue).toArray());
        }

        // For nested shapes, find and remove the mode
        ModeRange range = findModeRange(modeIndex);

        List<Long> newFlat = new ArrayList<>();
        List<Integer> newParent = new ArrayList<>();

        // Add dimensions before the removed mode
        for (int i = 0; i < range.start; i++) {
            newFlat.add(flat[i]);
            newParent.add(parent[i]);
        }

        // Skip the mode (range.start to range.end)

        // Add dimensions after the removed mode, adjusting parent indices
        int offset = range.start - range.end;
        for (int i = range.end; i < flat.length; i++) {
            newFlat.add(flat[i]);
            int p = parent[i];
            newParent.add(p == -1 ? -1 : (p < range.start ? p : p + offset));
        }

        return buildShape(newFlat, newParent);
    }

    @Override
    public Shape permute(int... _modeIndices) {
        if (_modeIndices.length != rank()) {
            throw new IllegalArgumentException("Permutation must have same length as rank: " + rank() + " vs " + _modeIndices.length);
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
        List<Integer> newParent = new ArrayList<>();

        for (int newPos = 0; newPos < axes.length; newPos++) {
            int oldPos = axes[newPos];
            ModeRange range = ranges[oldPos];
            int modeStart = newFlat.size();

            // Copy dimensions from the old mode
            for (int i = range.start; i < range.end; i++) {
                newFlat.add(flat[i]);
                if (i == range.start) {
                    newParent.add(-1);
                } else {
                    int p = parent[i];
                    if (p == -1) {
                        // Another root within the mode - child of mode start
                        newParent.add(modeStart);
                    } else {
                        // Remap parent relative to new mode start
                        newParent.add(modeStart + (p - range.start));
                    }
                }
            }
        }

        return buildShape(newFlat, newParent);
    }

    // Helper to find the range of flat indices for a mode
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

        // Find end: scan until next root or end of array
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

    // Helper to build shape from lists
    private Shape buildShape(List<Long> flatList, List<Integer> parentList) {
        if (flatList.isEmpty()) {
            return scalar();
        }

        long[] flatArray = new long[flatList.size()];
        int[] parentArray = new int[parentList.size()];
        for (int i = 0; i < flatList.size(); i++) {
            flatArray[i] = flatList.get(i);
            parentArray[i] = parentList.get(i);
        }

        // Check if all parents are -1 (flat structure)
        boolean allRoots = true;
        for (int p : parentArray) {
            if (p != -1) {
                allRoots = false;
                break;
            }
        }

        if (allRoots) {
            return ShapeImpl.of(flatArray);
        } else {
            return new ShapeImpl(flatArray, parentArray);
        }
    }
}
