package ai.qxotic.jota.runtime.hip;

import java.util.Objects;

public final class HipModule implements AutoCloseable {

    private final long handle;

    private HipModule(long handle) {
        this.handle = handle;
    }

    public static HipModule load(byte[] hsaco) {
        Objects.requireNonNull(hsaco, "hsaco");
        HipRuntime.requireAvailable();
        return new HipModule(HipRuntime.loadModule(hsaco));
    }

    public HipFunction function(String name) {
        Objects.requireNonNull(name, "name");
        HipRuntime.requireAvailable();
        return new HipFunction(HipRuntime.getFunction(handle, name));
    }

    @Override
    public void close() {
        HipRuntime.requireAvailable();
        HipRuntime.unloadModule(handle);
    }
}
