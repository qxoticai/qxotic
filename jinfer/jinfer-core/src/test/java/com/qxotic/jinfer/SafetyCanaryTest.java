package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The safety canary's mechanics, model-free: it stays silent while the backing arena lives, fires
 * with THE message the moment the arena closes (never resurrects), covers every segment-backed
 * tensor through the one {@code SegmentFloatTensor} implementation, and never fires for memory
 * whose scope cannot die. The per-family enforcement (a freed arena under a real model is a
 * teaching ISE, not a SIGSEGV) lives in {@code WeightsCanaryIT}.
 */
class SafetyCanaryTest {

    @Test
    void firesExactlyWhenTheArenaCloses() {
        Arena arena = Arena.ofShared();
        FloatTensor tensor = new F32FloatTensor(16, arena.allocate(16 * 4));
        tensor.safetyCanary(); // alive: silent
        arena.close();
        IllegalStateException e = assertThrows(IllegalStateException.class, tensor::safetyCanary);
        assertEquals(FloatTensor.FREED_MESSAGE, e.getMessage());
        // scopes never resurrect: the canary keeps firing
        assertThrows(IllegalStateException.class, tensor::safetyCanary);
    }

    @Test
    void quantizedTensorsShareTheOneImplementation() {
        // every GGML quantization extends SegmentFloatTensor, so ONE override guards them all -
        // pinned on Q4_0 (one 32-element block = 2 scale bytes + 16 nibble bytes)
        Arena arena = Arena.ofShared();
        FloatTensor tensor = new Q4_0FloatTensor(32, arena.allocate(18));
        tensor.safetyCanary();
        arena.close();
        IllegalStateException e = assertThrows(IllegalStateException.class, tensor::safetyCanary);
        assertTrue(e.getMessage().contains("freed"), e.getMessage());
    }

    @Test
    void undyingScopesNeverFire() {
        // the global scope cannot close, so a tensor over it is canary-silent forever - the
        // hand-built/raw case must never false-alarm
        MemorySegment global = MemorySegment.NULL.reinterpret(64);
        new F32FloatTensor(16, global).safetyCanary();
    }
}
