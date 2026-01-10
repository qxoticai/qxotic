package com.llm4j.jota;



import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public interface Shape2 {
//
//    int rank();
//
//    long dimension(int _axis);
//
//    long size();
//
//    default boolean hasZeroElements() {
//        return size() == 0;
//    }
//
//    default boolean hasOneElement() {
//        return size() == 1;
//    }
//
//    default boolean isScalar() {
//        return rank() == 0;
//    }
//
//    default boolean isVector() {
//        return rank() == 1;
//    }
//
//    default boolean isMatrix() {
//        return rank() == 2;
//    }
//
//    default boolean isVolume() {
//        return rank() == 3;
//    }
//
//    static Shape of(long... dimensions) {
//        return ShapeFactory.of(dimensions);
//    }
//
//    static Shape scalar() {
//        return ShapeFactory.scalar();
//    }
//
//    default long[] toArray() {
//        long[] array = new long[rank()];
//        for (int i = 0; i < rank(); i++) {
//            array[i] = dimension(i);
//        }
//        return array;
//    }
//
//    default Shape permute(int... permutationIndices) {
//        validatePermutation(permutationIndices, rank());
//        long[] newDims = new long[permutationIndices.length];
//        for (int i = 0; i < permutationIndices.length; i++) {
//            int srcIndex = permutationIndices[i];
//            newDims[i] = dimension(srcIndex);
//        }
//        return Shape.of(newDims);
//    }
//
//    private static void validatePermutation(int[] permutationIndices, int n) {
//        if (permutationIndices.length != n) {
//            throw new IllegalArgumentException("permute requires same number of dimensions");
//        }
//
//        boolean[] used = new boolean[permutationIndices.length];
//        for (int index : permutationIndices) {
//            if (index < 0 || index >= permutationIndices.length || used[index]) {
//                throw new IllegalArgumentException("Invalid permutation: " + Arrays.toString(permutationIndices));
//            }
//            used[index] = true;
//        }
//    }
//
//    default Shape swap(int _axis0, int _axis1) {
//        int axis0 = wrapAround(_axis0);
//        int axis1 = wrapAround(_axis1);
//        long[] newDims = toArray();
//        long tmp = newDims[axis0];
//        newDims[axis0] = newDims[axis1];
//        newDims[axis1] = tmp;
//        return Shape.of(newDims);
//    }
//
//    default Shape remove(int... _axes) {
//        if (_axes.length == 0) {
//            return this;
//        }
//        assert Arrays.stream(_axes).map(this::wrapAround).distinct().count() == _axes.length;
//        Set<Integer> toRemoveSet = Arrays.stream(_axes).map(this::wrapAround).boxed().collect(Collectors.toSet());
//        long[] newDims = IntStream.range(0, rank())
//                .filter(axis -> !toRemoveSet.contains(axis))
//                .mapToLong(this::dimension)
//                .toArray();
//        return Shape.of(newDims);
//    }
//
//    default Shape keep(int... _axes) {
//        assert Arrays.stream(_axes).map(this::wrapAround).distinct().count() == _axes.length;
//        Set<Integer> toKeepSet = Arrays.stream(_axes).map(this::wrapAround).boxed().collect(Collectors.toSet());
//        long[] newDims = IntStream.range(0, rank())
//                .filter(toKeepSet::contains)
//                .mapToLong(this::dimension)
//                .toArray();
//        return Shape.of(newDims);
//    }
//
//    default Shape append(long... dims) {
//        long[] newDims = Arrays.copyOf(toArray(), rank() + dims.length);
//        System.arraycopy(dims, 0, newDims, rank(), dims.length);
//        return Shape.of(newDims);
//    }
//
//    default Shape subShape(int fromInclusive, int toExclusive) {
//        if (fromInclusive == 0 && toExclusive == rank()) {
//            return this;
//        }
//        if (toExclusive > rank()) {
//            throw new ArrayIndexOutOfBoundsException();
//        }
//        return Shape.of(Arrays.copyOfRange(toArray(), fromInclusive, toExclusive));
//    }
//
//    default Shape replace(int _axis, long newDimension) {
//        int axis = wrapAround(_axis);
//        if (dimension(axis) == newDimension) {
//            return this;
//        }
//        long[] newDims = toArray();
//        newDims[axis] = newDimension;
//        return Shape.of(newDims);
//    }
//
//    default Shape prefix(int count) {
//        return subShape(0, count);
//    }
//
//    default Shape suffix(int count) {
//        return subShape(rank() - count, rank());
//    }
//
//    default Shape squeezeAll() {
//        int[] removeAxes = IntStream.range(0, rank()).filter(i -> dimension(i) == 1).toArray();
//        return remove(removeAxes);
//    }
//
//    default Shape squeeze(int _axis) {
//        if (dimension(_axis) != 1) {
//            throw new IllegalArgumentException("dimension != 1 at axis " + _axis);
//        }
//        return remove(_axis);
//    }
//
//    default Shape insert(int axis_, long dimension) {
//        int axis = wrapAround(axis_, rank() + 1);
//        if (dimension < 0) {
//            throw new IllegalArgumentException("dimension must be non-negative, got: " + dimension);
//        }
//        long[] newDims = new long[rank() + 1];
//        long[] oldDims = toArray();
//        System.arraycopy(oldDims, 0, newDims, 0, axis);
//        newDims[axis] = dimension;
//        System.arraycopy(oldDims, axis, newDims, axis + 1, rank() - axis);
//        return Shape.of(newDims);
//    }
//
//    default Shape unsqueeze(int axis_) {
//        return insert(axis_, 1);
//    }
//
//    static boolean sameAs(Shape a, Shape b) {
//        if (a.rank() != b.rank()) {
//            return false;
//        }
//        for (int i = 0; i < a.rank(); ++i) {
//            if (a.dimension(i) != b.dimension(i)) {
//                return false;
//            }
//        }
//        return true;
//    }
//
//    default int wrapAround(int _index) {
//        return wrapAround(_index, rank());
//    }
//
//    static int wrapAround(int _index, int rank) {
//        assert rank >= 0;
//        int index = _index >= 0 ? _index : _index + rank;
//        if (index < 0 || index >= rank) {
//            throw new IllegalArgumentException("wrap-around index out of bounds");
//        }
//        return index;
//    }
}

