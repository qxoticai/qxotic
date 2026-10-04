package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** No model needed: the weights arena is freed even when the state's cleanup throws. */
class CloseOrderTest {

    @Test
    void aFailingStateCloseStillFreesTheArena() {
        Arena arena = Arena.ofShared();
        RuntimeState failing =
                new RuntimeState() {
                    @Override
                    protected void releaseResources() {
                        throw new IllegalStateException("release failed");
                    }
                };
        assertThrows(
                IllegalStateException.class,
                () -> JinferEmbeddingModel.closeStateThenArena(failing, arena));
        assertFalse(arena.scope().isAlive(), "the arena leaked behind a failing state close");
    }
}
