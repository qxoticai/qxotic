package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryView;

public final class Indexing {

    private Indexing() {}

    public static long[] linearToCoord(Shape shape, long linearIndex) {
        long[] dims = shape.toArray();
        int rank = dims.length;
        if (rank == 0) {
            if (linearIndex != 0) {
                throw new IllegalArgumentException("Linear index out of bounds for scalar shape");
            }
            return new long[0];
        }
        long size = shape.size();
        if (linearIndex < 0 || linearIndex >= size) {
            throw new IllegalArgumentException("Linear index out of bounds: " + linearIndex);
        }

        long[] coord = new long[rank];
        long remaining = linearIndex;
        for (int i = rank - 1; i >= 0; i--) {
            long dim = dims[i];
            coord[i] = remaining % dim;
            remaining /= dim;
        }
        return coord;
    }

    public static long coordToLinear(Shape shape, long... coord) {
        long[] dims = shape.toArray();
        if (coord.length != dims.length) {
            throw new IllegalArgumentException(
                    "Coordinate flat rank "
                            + coord.length
                            + " does not match shape flat rank "
                            + dims.length);
        }
        if (dims.length == 0) {
            return 0;
        }

        long linear = 0;
        for (int i = 0; i < dims.length; i++) {
            long dim = dims[i];
            long value = coord[i];
            if (value < 0 || value >= dim) {
                throw new IllegalArgumentException(
                        "Coordinate out of bounds at axis " + i + ": " + value);
            }
            linear = Math.multiplyExact(linear, dim) + value;
        }
        return linear;
    }

    public static long coordToOffset(Stride stride, long... coord) {
        int flatRank = stride.flatRank();
        if (coord.length != flatRank) {
            throw new IllegalArgumentException(
                    "Coordinate flat rank "
                            + coord.length
                            + " does not match stride flat rank "
                            + flatRank);
        }
        long offset = 0;
        for (int i = 0; i < flatRank; i++) {
            offset += coord[i] * stride.flatAt(i);
        }
        return offset;
    }

    public static long coordToOffset(MemoryView<?> view, long... coord) {
        Stride stride = view.byteStride();
        int flatRank = stride.flatRank();
        if (coord.length != flatRank) {
            throw new IllegalArgumentException(
                    "Coordinate flat rank "
                            + coord.length
                            + " does not match stride flat rank "
                            + flatRank);
        }
        long offset = view.byteOffset();
        for (int i = 0; i < flatRank; i++) {
            offset += coord[i] * stride.flatAt(i);
        }
        return offset;
    }

    public static long linearToOffset(
            Shape shape, Stride stride, DataType dataType, long linearIndex) {
        long[] dims = shape.toArray();
        long[] strides = stride.toArray();
        if (dims.length != strides.length) {
            throw new IllegalArgumentException(
                    "Shape flat rank "
                            + dims.length
                            + " does not match stride flat rank "
                            + strides.length);
        }
        if (dims.length == 0) {
            if (linearIndex != 0) {
                throw new IllegalArgumentException("Linear index out of bounds for scalar shape");
            }
            return 0;
        }

        long size = shape.size();
        if (linearIndex < 0 || linearIndex >= size) {
            throw new IllegalArgumentException("Linear index out of bounds: " + linearIndex);
        }

        long elementOffset = 0;
        long remaining = linearIndex;
        for (int i = dims.length - 1; i >= 0; i--) {
            long dim = dims[i];
            long coord = remaining % dim;
            remaining /= dim;
            elementOffset += coord * strides[i];
        }
        return dataType.byteSizeFor(elementOffset);
    }

    public static long linearToOffset(Layout layout, DataType dataType, long linearIndex) {
        return linearToOffset(layout.shape(), layout.stride(), dataType, linearIndex);
    }

    public static long linearToOffset(MemoryView<?> view, long linearIndex) {
        long relativeOffset =
                linearToOffset(view.shape(), view.stride(), view.dataType(), linearIndex);
        return view.byteOffset() + relativeOffset;
    }
}
