package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.runtime.KernelCachePaths;
import java.nio.file.Path;

/** Cache layout for Mojo-generated source and binary artifacts. */
public final class MojoCachePaths {

    private MojoCachePaths() {}

    public static Path cacheRoot() {
        return KernelCachePaths.versionRoot().resolve("mojo");
    }

    public static Path sourceRoot() {
        return cacheRoot().resolve("sources");
    }

    public static Path lirSourceRoot() {
        return sourceRoot().resolve("lir");
    }

    public static Path lirKernelDir(String cacheKey) {
        return lirSourceRoot().resolve(cacheKey);
    }

    public static Path lirSourcePath(String cacheKey) {
        return lirKernelDir(cacheKey).resolve("kernel.mojo");
    }

    public static Path lirWrapperPath(String cacheKey) {
        return lirKernelDir(cacheKey).resolve("kernel.wrapper.mojo");
    }

    public static Path lirAsmPath(String cacheKey) {
        return lirKernelDir(cacheKey).resolve("kernel.mojo.s");
    }

    public static Path lirBridgeSourcePath(String cacheKey) {
        return lirSourceRoot().resolve(cacheKey + ".hip");
    }

    public static Path lirBinaryPath(String cacheKey) {
        return lirKernelDir(cacheKey).resolve("kernel.elf");
    }

    public static Path lirEntryPath(String cacheKey) {
        return lirKernelDir(cacheKey).resolve("kernel.entry");
    }
}
