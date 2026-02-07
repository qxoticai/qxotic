package ai.qxotic.jota.ir.base;

import ai.qxotic.jota.DataType;

/**
 * Base interface for all IR nodes across all IR levels (IR-T, IR-L, IR-K). This provides a unified
 * type system for the three-level IR hierarchy.
 */
public interface IRNode {

    /** Returns the data type of this node. */
    DataType dataType();
}
