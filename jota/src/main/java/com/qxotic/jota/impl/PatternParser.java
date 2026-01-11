package com.qxotic.jota.impl;

/**
 * Utility class for parsing nested tuple patterns.
 * Patterns specify the nesting structure using bracket notation.
 *
 * Examples:
 * - "[]" - scalar (no elements)
 * - "[_]" - singleton (one element)
 * - "[a, b, c]" - flat (three elements)
 * - "[batch, [N, M]]" - nested (rank 2, flatRank 3)
 */
final class PatternParser {

    private PatternParser() {
        // Utility class
    }

    /**
     * Parse a pattern string and return the parent array for the nested structure.
     *
     * @param pattern the pattern string (e.g., "[a, [b, c]]")
     * @param expectedElements the expected number of elements
     * @param elementTypeName the name of the element type for error messages (e.g., "dimension", "stride")
     * @return the parent array describing the nesting structure
     * @throws IllegalArgumentException if the pattern is malformed
     */
    static int[] parsePattern(String pattern, int expectedElements, String elementTypeName) {
        pattern = pattern.trim();
        if (!pattern.startsWith("[") || !pattern.endsWith("]")) {
            throw new IllegalArgumentException("Pattern must start with '[' and end with ']'");
        }

        int[] parent = new int[expectedElements];
        int[] elementIndex = {0};

        parseLevel(pattern, 1, pattern.length() - 1, -1, parent, elementIndex, elementTypeName);

        if (elementIndex[0] != expectedElements) {
            throw new IllegalArgumentException(
                "Pattern structure expects " + elementIndex[0] + " " + elementTypeName + "s but got " + expectedElements);
        }

        validateParentArray(parent);
        return parent;
    }

    /**
     * Validate the parent array structure for correctness.
     *
     * @param parent the parent array to validate
     * @throws IllegalArgumentException if the parent array is invalid
     */
    static void validateParentArray(int[] parent) {
        if (parent == null) {
            throw new IllegalArgumentException("Parent array cannot be null");
        }

        for (int i = 0; i < parent.length; i++) {
            int parentIndex = parent[i];

            // Parent must be -1 (top-level) or a valid index before current position
            if (parentIndex < -1) {
                throw new IllegalArgumentException(
                    "Invalid parent index at position " + i + ": " + parentIndex + " (must be >= -1)");
            }

            if (parentIndex >= i) {
                throw new IllegalArgumentException(
                    "Invalid parent index at position " + i + ": " + parentIndex +
                    " (parent must come before child, i.e., parent[i] < i)");
            }
        }
    }

    private static void parseLevel(String pattern, int start, int end, int parentIndex, int[] parent, int[] elementIndex, String elementTypeName) {
        // Handle empty brackets case (e.g., "[]" or just whitespace)
        int checkPos = start;
        while (checkPos < end && Character.isWhitespace(pattern.charAt(checkPos))) {
            checkPos++;
        }
        if (checkPos >= end) {
            // Empty brackets - valid only at top level for scalar
            return;
        }

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

                // The first element inside the nested brackets will be at elementIndex[0]
                int startIndex = elementIndex[0];
                if (elementIndex[0] >= parent.length) {
                    throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided " + elementTypeName + "s");
                }

                // Parse nested content - elements inside have startIndex as their logical parent
                parseLevel(pattern, nestedStart, nestedEnd, startIndex, parent, elementIndex, elementTypeName);

                // Validate the nested structure
                int elementsInNested = elementIndex[0] - startIndex;
                if (elementsInNested == 0) {
                    throw new IllegalArgumentException(
                        "Empty nested brackets [[]] are not allowed. " +
                        "Use [] for scalar or provide " + elementTypeName + "s inside brackets.");
                }
                if (elementsInNested == 1) {
                    throw new IllegalArgumentException(
                        "Single-element nested brackets like [[_]] are not normalized. " +
                        "Use [_] instead of [[_]].");
                }

                // Set the first element's parent to current parentIndex
                if (startIndex < parent.length) {
                    parent[startIndex] = parentIndex;
                }

                expectingElement = false;
            } else if (ch == ',') {
                // Empty identifiers are not allowed
                if (expectingElement) {
                    throw new IllegalArgumentException(
                        "Empty identifiers are not allowed in pattern. " +
                        "Each " + elementTypeName + " must have a name (e.g., use '_' or '" + elementTypeName + "' for placeholders)");
                }
                expectingElement = true;
                i++;
            } else if (ch != ']' && i < end) {
                // Found identifier (element name) - skip until comma or bracket
                while (i < end && pattern.charAt(i) != ',' && pattern.charAt(i) != '[' && pattern.charAt(i) != ']') {
                    i++;
                }

                // Record this element
                if (elementIndex[0] >= parent.length) {
                    throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided " + elementTypeName + "s");
                }
                parent[elementIndex[0]] = parentIndex;
                elementIndex[0]++;
                expectingElement = false;
            } else {
                break;
            }
        }
    }
}
