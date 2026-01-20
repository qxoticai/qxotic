package ai.qxotic.jota.tensor;

import java.util.Objects;
import java.util.function.Supplier;

public final class TensorOpsContext {

    private static final ThreadLocal<TensorOps> CURRENT = new ThreadLocal<>();

    private TensorOpsContext() {}

    public static TensorOps current() {
        return CURRENT.get();
    }

    public static TensorOps require() {
        TensorOps ops = CURRENT.get();
        if (ops == null) {
            throw new IllegalStateException("No TensorOps bound to the current thread");
        }
        return ops;
    }

    public static void set(TensorOps ops) {
        CURRENT.set(Objects.requireNonNull(ops));
    }

    public static void clear() {
        CURRENT.remove();
    }

    public static <T> T with(TensorOps ops, Supplier<T> action) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(action, "action");
        TensorOps previous = CURRENT.get();
        CURRENT.set(ops);
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
