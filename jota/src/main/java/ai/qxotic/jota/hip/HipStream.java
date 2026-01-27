package ai.qxotic.jota.hip;

public final class HipStream implements AutoCloseable {

    private final long handle;

    private HipStream(long handle) {
        this.handle = handle;
    }

    public static HipStream create() {
        HipRuntime.requireAvailable();
        return new HipStream(HipRuntime.createStream());
    }

    long handle() {
        return handle;
    }

    @Override
    public void close() {
        HipRuntime.requireAvailable();
        HipRuntime.destroyStream(handle);
    }
}
