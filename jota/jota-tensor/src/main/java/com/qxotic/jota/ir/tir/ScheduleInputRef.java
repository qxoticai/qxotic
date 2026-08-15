package com.qxotic.jota.ir.tir;

import java.util.Objects;

/** Reference to a value consumed by a scheduled kernel step. */
public sealed interface ScheduleInputRef
        permits ScheduleInputRef.TensorInputRef,
                ScheduleInputRef.ScalarInputRef,
                ScheduleInputRef.ProducedValueRef {

    record TensorInputRef(int inputId) implements ScheduleInputRef {
        public TensorInputRef {
            if (inputId < 0) {
                throw new IllegalArgumentException(
                        "Tensor input id must be non-negative, got: " + inputId);
            }
        }
    }

    record ScalarInputRef(int inputId) implements ScheduleInputRef {
        public ScalarInputRef {
            if (inputId < 0) {
                throw new IllegalArgumentException(
                        "Scalar input id must be non-negative, got: " + inputId);
            }
        }
    }

    record ProducedValueRef(ValueId valueId) implements ScheduleInputRef {
        public ProducedValueRef {
            Objects.requireNonNull(valueId, "valueId");
        }
    }
}
