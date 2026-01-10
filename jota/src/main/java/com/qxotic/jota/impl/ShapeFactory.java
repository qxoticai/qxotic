package com.qxotic.jota.impl;

import com.qxotic.jota.Shape;

public final class ShapeFactory {

    public static Shape flat(long... dims) {
        return ShapeImpl.flat(dims);
    }

    public static Shape pattern(String pattern, long... dims) {
        if (dims.length == 0) {
            return ShapeImpl.scalar();
        }

        try {
            int[] parent = parseNestingPattern(pattern, dims.length);
            return ShapeImpl.of(dims, parent);
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException("Pattern has more elements than provided dimensions", e);
        }
    }

    public static Shape template(NestedTuple<?> template, long... dims) {
        if (template instanceof NestedTupleImpl<?> impl) {
            if (impl.flatRank() != dims.length) {
                throw new IllegalArgumentException(
                    "Template has " + impl.flatRank() + " dimensions but " + dims.length + " were provided");
            }
            if (dims.length == 0) {
                return ShapeImpl.scalar();
            }
            return ShapeImpl.of(dims, impl.parent);
        }
        throw new IllegalArgumentException("Unsupported NestedTuple implementation");
    }

    public static Shape of(Object... elements) {
        if (elements.length == 0) {
            return ShapeImpl.scalar();
        }

        // Composition: of(Number/Shape... elements)
        return ShapeImpl.nested(elements);
    }

    public static Shape scalar() {
        return ShapeImpl.scalar();
    }

    private static int[] parseNestingPattern(String pattern, int expectedDims) {
        pattern = pattern.trim();
        if (!pattern.startsWith("[") || !pattern.endsWith("]")) {
            throw new IllegalArgumentException("Pattern must start with '[' and end with ']'");
        }

        int[] parent = new int[expectedDims];
        int[] dimIndex = {0};

        parseLevel(pattern, 1, pattern.length() - 1, -1, parent, dimIndex);

        if (dimIndex[0] != expectedDims) {
            throw new IllegalArgumentException(
                "Pattern structure expects " + dimIndex[0] + " dimensions but got " + expectedDims);
        }

        return parent;
    }

    private static void parseLevel(String pattern, int start, int end, int parentIndex, int[] parent, int[] dimIndex) {
        int i = start;
        boolean expectingElement = true;

        while (i <= end) {
            // Skip whitespace
            while (i < end && Character.isWhitespace(pattern.charAt(i))) {
                i++;
            }

            char ch = (i < end) ? pattern.charAt(i) : ','; // Treat end as comma

            if (ch == '[') {
                // Found nested structure
                int bracketCount = 1;
                int nestedStart = i + 1;
                i++;
                while (i < end && bracketCount > 0) {
                    if (pattern.charAt(i) == '[') bracketCount++;
                    else if (pattern.charAt(i) == ']') bracketCount--;
                    i++;
                }
                int nestedEnd = i - 1;

                // The first element inside the nested brackets will be at dimIndex[0]
                int startIndex = dimIndex[0];
                if (dimIndex[0] >= parent.length) {
                    throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided dimensions");
                }

                // Parse nested content - elements inside have startIndex as their logical parent
                parseLevel(pattern, nestedStart, nestedEnd, startIndex, parent, dimIndex);

                // Set the first element's parent to current parentIndex
                if (startIndex < parent.length) {
                    parent[startIndex] = parentIndex;
                }

                expectingElement = false;
            } else if (ch == ',') {
                // If we were expecting an element and hit comma, it's an empty identifier
                if (expectingElement) {
                    if (dimIndex[0] >= parent.length) {
                        throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided dimensions");
                    }
                    parent[dimIndex[0]] = parentIndex;
                    dimIndex[0]++;
                }
                expectingElement = true;
                i++;
            } else if (ch != ']' && i < end) {
                // Found identifier (dimension name) - skip until comma or bracket
                while (i < end && pattern.charAt(i) != ',' && pattern.charAt(i) != '[' && pattern.charAt(i) != ']') {
                    i++;
                }

                // Record this dimension
                if (dimIndex[0] >= parent.length) {
                    throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided dimensions");
                }
                parent[dimIndex[0]] = parentIndex;
                dimIndex[0]++;
                expectingElement = false;
            } else {
                break;
            }
        }
    }
}
