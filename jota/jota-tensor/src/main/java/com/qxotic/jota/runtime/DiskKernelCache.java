package com.qxotic.jota.runtime;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import java.nio.file.Path;

public record DiskKernelCache(Path root, String packageName) implements KernelCache {

    private static final String DEFAULT_PACKAGE = "com.qxotic.jota.runtime.jit";

    public static DiskKernelCache defaultCache(Device device) {
        return new DiskKernelCache(KernelCachePaths.deviceRoot(device), DEFAULT_PACKAGE);
    }

    public static DiskKernelCache defaultCache(DeviceType deviceType) {
        return new DiskKernelCache(KernelCachePaths.deviceRoot(deviceType), DEFAULT_PACKAGE);
    }

    public DiskKernelCache(Path root) {
        this(root, DEFAULT_PACKAGE);
    }

    @Override
    public KernelCacheEntry entryFor(KernelCacheKey key) {
        String className = "Kernel_" + key.value().substring(0, 16);
        Path directory = root.resolve(key.value());
        Path sourcePath = directory.resolve(className + ".java");
        Path classOutputDir = directory.resolve("classes");
        Path classFilePath =
                classOutputDir.resolve(packageName.replace('.', '/')).resolve(className + ".class");
        return new KernelCacheEntry(
                key, packageName, className, directory, sourcePath, classOutputDir, classFilePath);
    }
}
