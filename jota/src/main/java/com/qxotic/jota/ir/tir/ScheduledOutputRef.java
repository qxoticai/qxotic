package com.qxotic.jota.ir.tir;

/** Reference to the final output produced by a scheduled program. */
public sealed interface ScheduledOutputRef
        permits ScheduledOutputRef.ValueOutput,
                ScheduledOutputRef.TensorInputOutput,
                ScheduledOutputRef.ScalarInputOutput {

    record ValueOutput(ValueId valueId) implements ScheduledOutputRef {
        public ValueOutput {
            java.util.Objects.requireNonNull(valueId, "valueId");
        }
    }

    record TensorInputOutput(int inputId) implements ScheduledOutputRef {
        public TensorInputOutput {
            if (inputId < 0) {
                throw new IllegalArgumentException(
                        "Tensor input output id must be non-negative, got: " + inputId);
            }
        }
    }

    record ScalarInputOutput(int inputId) implements ScheduledOutputRef {
        public ScalarInputOutput {
            if (inputId < 0) {
                throw new IllegalArgumentException(
                        "Scalar input output id must be non-negative, got: " + inputId);
            }
        }
    }
}
