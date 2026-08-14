package com.qxotic.jinfer.x.llm;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.x.boundary.RuntimeState;
import org.junit.jupiter.api.Test;

/** The 1..8 depth contract lives at the interface's production door, not in port prose. */
class SpeculativeDecodingTest {

    private static final SpeculativeDecoding<RuntimeState> MUST_NOT_BE_ENTERED =
            new SpeculativeDecoding<>() {
                @Override
                public boolean speculationReady() {
                    return true;
                }

                @Override
                public SpeculationResult speculate(
                        RuntimeState state,
                        Sampler sampler,
                        Generator.Constraints constraints,
                        int depth,
                        Generator.GenerationListener listener,
                        SpeculationAudit audit) {
                    throw new AssertionError("the bounds check must fire before the port runs");
                }
            };

    @Test
    void depthOutsideTheContractIsRefusedBeforeThePortRuns() {
        assertThrows(
                IllegalArgumentException.class,
                () -> MUST_NOT_BE_ENTERED.speculate(null, null, null, 0, null));
        assertThrows(
                IllegalArgumentException.class,
                () -> MUST_NOT_BE_ENTERED.speculate(null, null, null, 9, null));
    }
}
