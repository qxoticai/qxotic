package ai.qxotic.jota.runtime.hip;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

final class HipKernelSourceLoader {

    private HipKernelSourceLoader() {}

    static byte[] load(String path) {
        try {
            return Files.readAllBytes(Path.of(path));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read HSACO: " + path, e);
        }
    }
}
