package com.qxotic.jota.tensor;

import com.qxotic.jota.random.RandomKey;
import java.util.concurrent.atomic.AtomicLong;

final class TensorRandomState {
    private static final long DEFAULT_SEED = 0x5eed5eedL;

    private static final ThreadLocal<State> LOCAL =
            ThreadLocal.withInitial(
                    () -> new State(RandomKey.of(DEFAULT_SEED), new AtomicLong(0L)));

    private TensorRandomState() {}

    static void manualSeed(long seed) {
        LOCAL.set(new State(RandomKey.of(seed), new AtomicLong(0L)));
    }

    static RandomKey nextKey() {
        State state = LOCAL.get();
        long stream = state.counter().getAndIncrement();
        return state.baseKey().split(stream);
    }

    private record State(RandomKey baseKey, AtomicLong counter) {}
}
