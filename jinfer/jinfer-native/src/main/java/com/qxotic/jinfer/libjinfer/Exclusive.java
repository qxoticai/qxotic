package com.qxotic.jinfer.libjinfer;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One call at a time on a state or a stream, as the header says: a second, concurrent one is
 * JINFER_EBUSY rather than a race. Entered by the entry points, never held across calls.
 */
final class Exclusive {
    private final AtomicBoolean busy = new AtomicBoolean();
    private final String what;

    Exclusive(String what) {
        this.what = what;
    }

    void enter() {
        if (!busy.compareAndSet(false, true))
            throw new Errors.Failure(
                    Errors.EBUSY, "the " + what + " is in a call on another thread");
    }

    void leave() {
        busy.set(false);
    }
}
