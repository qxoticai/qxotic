package com.qxotic.jota.runtime.metal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class MetalKernelSourceLoader {

    private MetalKernelSourceLoader() {}

    static byte[] load(String path) {
        try {
            return Files.readAllBytes(Path.of(path));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read metallib: " + path, e);
        }
    }
}
