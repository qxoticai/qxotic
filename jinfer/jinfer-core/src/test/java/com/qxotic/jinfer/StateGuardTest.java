package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ConcurrentModificationException;
import org.junit.jupiter.api.Test;

class StateGuardTest {

    private static RuntimeState state() {
        return new RuntimeState() {
            public int contextCapacity() {
                return 0;
            }

            public int batchCapacity() {
                return 0;
            }

            public int position() {
                return 0;
            }

            public int outputCount() {
                return 0;
            }
        };
    }

    @Test
    void secondThreadIsRejectedWhileHeld() throws Exception {
        RuntimeState s = state();
        StateGuard.claim(s);
        Thread t =
                new Thread(
                        () ->
                                assertThrows(
                                        ConcurrentModificationException.class,
                                        () -> StateGuard.claim(s)));
        t.start();
        t.join();
        StateGuard.release(s);
        // released: any thread may claim again
        Thread t2 = new Thread(() -> assertDoesNotThrow(() -> StateGuard.claim(s)));
        t2.start();
        t2.join();
    }

    @Test
    void reclaimBySameThreadIsIdempotentAndIndependentStatesDoNotInterfere() {
        RuntimeState a = state();
        RuntimeState b = state();
        StateGuard.claim(a);
        assertDoesNotThrow(() -> StateGuard.claim(a)); // re-entrant for the holder
        assertDoesNotThrow(() -> StateGuard.claim(b)); // a different state is a different pipeline
        StateGuard.release(a);
        StateGuard.release(b);
    }
}
