package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Environment;
import java.util.Objects;
import java.util.function.Supplier;

public final class TensorOpsContext {

    private static final ScopedValue<TensorOps> CURRENT = ScopedValue.newInstance();

    private TensorOpsContext() {}

    public static TensorOps current() {
        return CURRENT.isBound() ? CURRENT.get() : null;
    }

    public static TensorOps require() {
        if (CURRENT.isBound()) {
            return CURRENT.get();
        }
        return EnvironmentOps.resolve(Environment.current());
    }

    public static <T> T withIR(Supplier<T> supplier) {
        return with(new IRTensorOps(), supplier);
    }

    public static <T> T with(TensorOps ops, Supplier<T> action) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(action, "action");
        try {
            return ScopedValue.where(CURRENT, ops).call(action::get);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Scoped TensorOps action failed", e);
        }
    }
}
