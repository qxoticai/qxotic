package com.qxotic.jota.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PatternValidationTest {

    @Test
    void testValidateNullParentArray() {
        // Null parent arrays are actually valid (they indicate flat structure)
        // So validateParentArray should accept null
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.validateParentArray(null);
        }, "validateParentArray still rejects null for explicit validation");
    }

    @Test
    void testValidateParentIndexTooSmall() {
        // Parent index -2 is invalid (must be >= -1)
        int[] parent = new int[]{-1, -2, 1};
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.validateParentArray(parent);
        }, "Parent index < -1 should be rejected");
    }

    @Test
    void testValidateParentIndexEqualToCurrentIndex() {
        // Parent cannot be equal to current index
        int[] parent = new int[]{-1, 1, -1};
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.validateParentArray(parent);
        }, "Parent index equal to current index should be rejected");
    }

    @Test
    void testValidateParentIndexGreaterThanCurrentIndex() {
        // Parent cannot be after child (parent[1] = 2, but child is at index 1)
        int[] parent = new int[]{-1, 2, -1};
        assertThrows(IllegalArgumentException.class, () -> {
            PatternParser.validateParentArray(parent);
        }, "Parent index > current index should be rejected (parent must come before child)");
    }

    @Test
    void testValidateCircularReference() {
        // While not technically circular (parent must be < i), this tests the constraint
        int[] parent = new int[]{-1, 0, 1};
        // This should be valid: parent[0]=-1, parent[1]=0, parent[2]=1
        assertDoesNotThrow(() -> {
            PatternParser.validateParentArray(parent);
        });
    }

    @Test
    void testValidateFlatStructure() {
        // All elements at top level
        int[] parent = new int[]{-1, -1, -1};
        assertDoesNotThrow(() -> {
            PatternParser.validateParentArray(parent);
        });
    }

    @Test
    void testValidateNestedStructure() {
        // Valid nested structure: [0, [1, 2]]
        // parent[0] = -1 (top level)
        // parent[1] = -1 (top level)
        // parent[2] = 1 (nested under element 1)
        int[] parent = new int[]{-1, -1, 1};
        assertDoesNotThrow(() -> {
            PatternParser.validateParentArray(parent);
        });
    }

    @Test
    void testValidateDeeplyNestedStructure() {
        // Valid deeply nested: [0, [1, [2, 3]]]
        // parent[0] = -1
        // parent[1] = -1
        // parent[2] = 1
        // parent[3] = 2
        int[] parent = new int[]{-1, -1, 1, 2};
        assertDoesNotThrow(() -> {
            PatternParser.validateParentArray(parent);
        });
    }

    @Test
    void testValidateEmptyParentArray() {
        // Empty array (scalar)
        int[] parent = new int[]{};
        assertDoesNotThrow(() -> {
            PatternParser.validateParentArray(parent);
        });
    }
}
