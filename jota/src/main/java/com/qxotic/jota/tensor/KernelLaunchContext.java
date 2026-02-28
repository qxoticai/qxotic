package com.qxotic.jota.tensor;

import java.util.Objects;
import java.util.concurrent.Executor;

public final class KernelLaunchContext {

    private static final KernelLaunchContext DISABLED =
            new KernelLaunchContext(
                    false, null, 16_384, Math.max(1, Runtime.getRuntime().availableProcessors()));

    private final boolean parallelEnabled;
    private final Executor executor;
    private final int minTripCount;
    private final int targetTasks;

    private KernelLaunchContext(
            boolean parallelEnabled, Executor executor, int minTripCount, int targetTasks) {
        this.parallelEnabled = parallelEnabled;
        this.executor = executor;
        this.minTripCount = minTripCount;
        this.targetTasks = targetTasks;
    }

    public static KernelLaunchContext disabled() {
        return DISABLED;
    }

    public static KernelLaunchContext ofParallel(
            Executor executor, int minTripCount, int targetTasks) {
        Objects.requireNonNull(executor, "executor");
        if (minTripCount <= 0) {
            throw new IllegalArgumentException("minTripCount must be > 0");
        }
        if (targetTasks <= 0) {
            throw new IllegalArgumentException("targetTasks must be > 0");
        }
        return new KernelLaunchContext(true, executor, minTripCount, targetTasks);
    }

    public boolean parallelEnabled() {
        return parallelEnabled;
    }

    public Executor executor() {
        return executor;
    }

    public int minTripCount() {
        return minTripCount;
    }

    public int targetTasks() {
        return targetTasks;
    }
}
