package ai.qxotic.jota.impl;

/**
 * Utility class for parsing nested tuple patterns.
 * Patterns specify the nesting structure using bracket notation.
 *
 * Examples:
 * - "()" - scalar (no elements)
 * - "(_)" - singleton (one element)
 * - "(a, b, c)" - flat (three elements)
 * - "(batch, (N, M))" - nested (rank 2, flatRank 3)
 */
final class PatternParser {

    private PatternParser() {
        // Utility class
    }

    /**
     * Parse a pattern string and return the nest array for the nested structure.
     *
     * @param pattern the pattern string (e.g., "(a, (b, c))")
     * @param expectedElements the expected number of elements
     * @param elementTypeName the name of the element type for error messages (e.g., "dimension", "stride")
     * @return the nest array describing the nesting structure
     * @throws IllegalArgumentException if the pattern is malformed
     */
    static int[] parsePattern(String pattern, int expectedElements, String elementTypeName) {
        if (pattern == null) {
            throw new IllegalArgumentException("Pattern cannot be null");
        }
        char openChar = pattern.charAt(0);
        char closeChar = pattern.charAt(pattern.length() - 1);
        if (openChar != '(' || closeChar != ')') {
            throw new IllegalArgumentException("Pattern must start with '(' and end with ')'");
        }

        int[] nest = new int[expectedElements];
        int[] groupCounts = new int[pattern.length()];
        boolean[] expectingElement = new boolean[pattern.length()];
        int groupDepth = 0;
        int pendingOpens = 0;
        int lastLeafIndex = -1;
        int elementIndex = 0;
        boolean sawAny = false;

        groupCounts[0] = 0;
        expectingElement[0] = true;

        for (int index = 1; index < pattern.length() - 1; ) {
            char ch = pattern.charAt(index);

            if (Character.isWhitespace(ch)) {
                index++;
                continue;
            }

            if (ch == ',') {
                if (expectingElement[groupDepth]) {
                    throw new IllegalArgumentException(
                        "Empty identifiers are not allowed in pattern. " +
                        "Each " + elementTypeName + " must have a name (e.g., use '_' or '" + elementTypeName + "' for placeholders)");
                }
                expectingElement[groupDepth] = true;
                index++;
                continue;
            }

            if (ch == openChar) {
                if (!expectingElement[groupDepth]) {
                    throw new IllegalArgumentException("Missing ',' between elements in pattern");
                }
                pendingOpens++;
                groupDepth++;
                groupCounts[groupDepth] = 0;
                expectingElement[groupDepth] = true;
                index++;
                continue;
            }

            if (ch == closeChar) {
                if (groupDepth == 0) {
                    throw new IllegalArgumentException("Unbalanced '" + closeChar + "' in pattern");
                }
                if (expectingElement[groupDepth]) {
                    throw new IllegalArgumentException("Trailing ',' before ')' in pattern");
                }
                if (groupCounts[groupDepth] == 0) {
                    throw new IllegalArgumentException(
                        "Empty nested brackets are not allowed. " +
                        "Use " + openChar + closeChar + " for scalar or provide " + elementTypeName + "s inside brackets.");
                }
                if (groupCounts[groupDepth] == 1) {
                    throw new IllegalArgumentException(
                        "Single-element nested brackets are not normalized. " +
                        "Use " + openChar + "_" + closeChar + " instead.");
                }
                if (lastLeafIndex < 0) {
                    throw new IllegalArgumentException("Pattern must contain at least one " + elementTypeName);
                }

                nest[lastLeafIndex] -= 1;
                groupDepth--;
                groupCounts[groupDepth]++;
                expectingElement[groupDepth] = false;
                index++;
                continue;
            }

            if (!expectingElement[groupDepth]) {
                throw new IllegalArgumentException("Missing ',' between elements in pattern");
            }

            boolean hasTokenChar = false;
            while (index < pattern.length() - 1) {
                char current = pattern.charAt(index);
                if (current == ',' || current == openChar || current == closeChar) {
                    break;
                }
                if (!Character.isWhitespace(current)) {
                    hasTokenChar = true;
                }
                index++;
            }

            if (!hasTokenChar) {
                throw new IllegalArgumentException("Empty identifiers are not allowed in pattern");
            }
            if (elementIndex >= expectedElements) {
                throw new ArrayIndexOutOfBoundsException("Pattern has more elements than provided " + elementTypeName + "s");
            }

            nest[elementIndex] = pendingOpens;
            pendingOpens = 0;
            lastLeafIndex = elementIndex;
            elementIndex++;
            sawAny = true;
            groupCounts[groupDepth]++;
            expectingElement[groupDepth] = false;
        }

        if (expectingElement[groupDepth] && groupCounts[groupDepth] > 0) {
            throw new IllegalArgumentException("Trailing ',' before ')' in pattern");
        }
        if (groupDepth != 0) {
            throw new IllegalArgumentException("Unbalanced '(' in pattern");
        }
        if (!sawAny && expectedElements != 0) {
            throw new IllegalArgumentException(
                "Pattern structure expects 0 " + elementTypeName + "s but got " + expectedElements);
        }
        if (elementIndex != expectedElements) {
            throw new IllegalArgumentException(
                "Pattern structure expects " + elementIndex + " " + elementTypeName + "s but got " + expectedElements);
        }

        return nest;
    }

}
