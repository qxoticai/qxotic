package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import java.nio.file.Path;

public final class DiskKernelCache implements KernelCache {

    private static final String DEFAULT_PACKAGE = "ai.qxotic.jota.tensor.jit";
    private static final Path DEFAULT_ROOT = Path.of("__kernels").resolve(Device.PANAMA.leafName());

    private final Path root;
    private final String packageName;

    public static DiskKernelCache defaultCache() {
        return new DiskKernelCache(DEFAULT_ROOT, DEFAULT_PACKAGE);
    }

    public DiskKernelCache(Path root) {
        this(root, DEFAULT_PACKAGE);
    }

    public DiskKernelCache(Path root, String packageName) {
        this.root = root;
        this.packageName = packageName;
    }

    @Override
    public Path root() {
        return root;
    }

    public String packageName() {
        return packageName;
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
