package ai.qxotic.jota.tensor;

public interface KernelExecutable extends AutoCloseable {

    void launch(LaunchConfig config, KernelArgs args, ExecutionStream stream);

    @Override
    default void close() {}
}
