package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.boundary.ContextState;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** The 1..8 depth contract lives at the interface's production door, not in port prose. */
class SpeculativeDecodingTest {

    private static final class State extends ContextState {
        State() {
            super(8, 8, new PanamaMemoryArena(Arena.ofAuto()), false);
        }

        @Override
        protected void clearHistory() {}
    }

    private static final SpeculativeDecoding<State> MUST_NOT_BE_ENTERED =
            new SpeculativeDecoding<>() {
                @Override
                public boolean speculationReady() {
                    return true;
                }

                @Override
                public SpeculationResult speculate(
                        State state,
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
