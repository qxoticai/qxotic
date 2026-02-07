package ai.qxotic.jota.ir.tir;

/** Represents a slice range for tensor slicing operations. */
public record SliceRange(long start, long end, long step) {

    public SliceRange {
        if (step == 0) {
            throw new IllegalArgumentException("step cannot be zero");
        }
    }

    /** Creates a slice range from start to end with step=1. */
    public static SliceRange of(long start, long end) {
        return new SliceRange(start, end, 1);
    }

    /** Creates a slice range that includes all elements. */
    public static SliceRange all() {
        return new SliceRange(0, Long.MAX_VALUE, 1);
    }

    /** Returns the number of elements in this slice range. */
    public long size() {
        return (end - start + step - 1) / step;
    }
}
