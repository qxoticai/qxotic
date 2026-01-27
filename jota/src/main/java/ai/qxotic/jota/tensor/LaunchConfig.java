package ai.qxotic.jota.tensor;

public record LaunchConfig(
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int blockDimX,
        int blockDimY,
        int blockDimZ,
        int sharedMemBytes,
        boolean autoLaunch) {

    public static LaunchConfig auto() {
        return new LaunchConfig(0, 0, 0, 0, 0, 0, 0, true);
    }
}
