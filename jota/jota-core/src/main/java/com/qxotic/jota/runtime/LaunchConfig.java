package com.qxotic.jota.runtime;

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

    public static LaunchConfig grid(int x) {
        return grid(x, 1, 1);
    }

    public static LaunchConfig grid(int x, int y) {
        return grid(x, y, 1);
    }

    public static LaunchConfig grid(int x, int y, int z) {
        return new LaunchConfig(x, y, z, 1, 1, 1, 0, false);
    }

    public LaunchConfig block(int x) {
        return block(x, 1, 1);
    }

    public LaunchConfig block(int x, int y) {
        return block(x, y, 1);
    }

    public LaunchConfig block(int x, int y, int z) {
        return new LaunchConfig(gridDimX, gridDimY, gridDimZ, x, y, z, sharedMemBytes, false);
    }

    public LaunchConfig sharedMem(int bytes) {
        return new LaunchConfig(
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, bytes, autoLaunch);
    }
}
