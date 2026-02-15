package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;

/**
 * Sealed interface for LIR inputs. An input can be either a buffer (tensor passed by pointer) or a
 * scalar (value passed directly).
 */
public sealed interface LIRInput permits BufferRef, ScalarInput {

    /** Returns the unique input ID. */
    int id();

    /** Returns the data type of this input. */
    DataType dataType();
}
