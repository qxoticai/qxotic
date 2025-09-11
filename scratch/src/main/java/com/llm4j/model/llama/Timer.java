package com.llm4j.model.llama;

import java.util.concurrent.TimeUnit;
import java.util.function.LongFunction;
import java.util.function.Supplier;

public interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(TimeUnit.MILLISECONDS, label);
    }

    System.Logger LOGGER = System.getLogger("Timer");

    static Timer log(TimeUnit timeUnit, String label) {
        return log(System.Logger.Level.INFO, timeUnit, label);
    }

    static Timer log(System.Logger.Level level, TimeUnit timeUnit, LongFunction<String> message) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                if (LOGGER.isLoggable(level)) {
                    long convertedUnits = timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS);
                    LOGGER.log(level, message.apply(convertedUnits));
                }
            }
        };
    }

    static Timer log(System.Logger.Level level, TimeUnit timeUnit, String label) {
        return log(level, timeUnit, elapsedUnits -> label + ": " + elapsedUnits + " " + timeUnit.toChronoUnit().name().toLowerCase());
    }

    static void measure(String label, ThrowingRunnable action) {
        measure(label, TimeUnit.MILLISECONDS, action);
    }

    static void measure(String label, TimeUnit timeUnit, ThrowingRunnable action) {
        try (Timer timer = Timer.log(timeUnit, label)) {
            action.run();
        }
    }

    @SuppressWarnings("unchecked")
    private static <E extends Throwable> RuntimeException uncheckedThrow(Throwable throwable) throws E {
        throw (E) throwable;
    }

    @FunctionalInterface
    interface ThrowingSupplier<T> extends Supplier<T> {

        @Override
        default T get() {
            try {
                return throwingGet();
            } catch (Throwable throwable) {
                throw uncheckedThrow(throwable);
            }
        }

        T throwingGet() throws Throwable;
    }

    @FunctionalInterface
    interface ThrowingRunnable extends Runnable {
        @Override
        default void run() {
            try {
                throwingRun();
            } catch (Throwable throwable) {
                throw uncheckedThrow(throwable);
            }
        }

        void throwingRun() throws Throwable;
    }
}
