package com.qxotic.jota.runtime.opencl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class OpenClKernelSourceLoader {

    private OpenClKernelSourceLoader() {}

    static byte[] load(String path) {
        try {
            return Files.readAllBytes(Path.of(path));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read OpenCL source: " + path, e);
        }
    }
}
