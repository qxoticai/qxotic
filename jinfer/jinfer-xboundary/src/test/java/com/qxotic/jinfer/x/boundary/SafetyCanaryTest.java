package com.qxotic.jinfer.x.boundary;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/**
 * The safety canary's mechanics, model-free. In x the canary is {@link BaseState#enter()}'s
 * arena-liveness check (one honest {@link com.qxotic.jinfer.x.PanamaMemoryArena#isAlive()} probe
 * covers every buffer the state allocated, whatever the dtype - the per-tensor canary of the old
 * FloatTensor tree is gone with the tensors): silent while the backing arena lives, firing with THE
 * message the moment it closes (never resurrects), never firing for scopes that cannot die. The
 * per-family enforcement (a freed arena under a real model is a teaching ISE, not a SIGSEGV) lives
 * in the weights-canary integration tests.
 */
class SafetyCanaryTest {

    @Test
    void firesExactlyWhenTheBorrowedArenaCloses() {
        Arena arena = Arena.ofShared();
        BaseStateLifecycleTest.ProbeState state = new BaseStateLifecycleTest.ProbeState(arena);
        state.enter(); // alive: silent
        state.exit();
        arena.close();
        IllegalStateException e = assertThrows(IllegalStateException.class, state::enter);
        assertEquals(BaseState.FREED_MESSAGE, e.getMessage());
        // scopes never resurrect: the canary keeps firing
        assertThrows(IllegalStateException.class, state::enter);
    }

    @Test
    void undyingScopesNeverFire() {
        // ofAuto/global cannot close, so a state over them is canary-silent forever
        BaseStateLifecycleTest.ProbeState auto =
                new BaseStateLifecycleTest.ProbeState(Arena.ofAuto());
        auto.enter();
        auto.exit();
        BaseStateLifecycleTest.ProbeState global =
                new BaseStateLifecycleTest.ProbeState(Arena.global());
        global.enter();
        global.exit();
    }
}
