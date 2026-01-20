package ai.qxotic.jota.tensor;

import java.util.Objects;
import java.util.function.Supplier;

public final class ComputeEngineContext {

    private static final ThreadLocal<ComputeEngine> CURRENT = new ThreadLocal<>();

    private ComputeEngineContext() {}

    public static ComputeEngine current() {
        return CURRENT.get();
    }

    public static ComputeEngine require() {
        ComputeEngine engine = CURRENT.get();
        if (engine == null) {
            throw new IllegalStateException("No ComputeEngine bound to the current thread");
        }
        return engine;
    }

    public static void set(ComputeEngine engine) {
        CURRENT.set(Objects.requireNonNull(engine));
    }

    public static void clear() {
        CURRENT.remove();
    }

    public static <T> T with(ComputeEngine engine, Supplier<T> action) {
        Objects.requireNonNull(engine, "engine");
        Objects.requireNonNull(action, "action");
        ComputeEngine previous = CURRENT.get();
        CURRENT.set(engine);
        try {
            return action.get();
        } finally {
            if (previous == null) {
                CURRENT.remove();
            } else {
                CURRENT.set(previous);
            }
        }
    }
}
