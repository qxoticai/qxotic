package com.qxotic.jinfer;

import java.util.Collections;
import java.util.ConcurrentModificationException;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Enforces the single-mutator contract on {@link RuntimeState}: one generation at a time per state.
 * The engine layers already serialize on their locks; this guard makes the LOW-LEVEL misuse loud -
 * two threads driving the same state would otherwise corrupt KV silently. Weak identity keys:
 * states never leak through the guard, and states do not override equals.
 */
public final class StateGuard {

    private static final Map<RuntimeState, Thread> ACTIVE =
            Collections.synchronizedMap(new WeakHashMap<>());

    private StateGuard() {}

    /** Claims {@code state} for the current thread; throws if another thread holds it. */
    public static void claim(RuntimeState state) {
        Thread prior = ACTIVE.putIfAbsent(state, Thread.currentThread());
        if (prior != null && prior != Thread.currentThread()) {
            throw new ConcurrentModificationException(
                    "model state is a single serial pipeline (one generation at a time; '"
                            + prior.getName()
                            + "' holds it) - for parallel pipelines create separate model"
                            + " instances/states");
        }
    }

    /** Releases {@code state}; a no-op when the current thread does not hold it. */
    public static void release(RuntimeState state) {
        ACTIVE.remove(state, Thread.currentThread());
    }
}
