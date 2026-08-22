package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Test;

class FfmpegTest {

    @Test
    void returnsBoundedProcessOutput() throws Exception {
        assertArrayEquals(
                "ok".getBytes(StandardCharsets.UTF_8),
                Ffmpeg.run(command("small"), null, Duration.ofSeconds(2), 16));
        IOException failure =
                assertThrows(
                        IOException.class,
                        () -> Ffmpeg.run(command("large"), null, Duration.ofSeconds(2), 1024));
        assertTrue(failure.getMessage().contains("exceeds"), failure.getMessage());
    }

    @Test
    void destroysAProcessAtItsDeadline() {
        long start = System.nanoTime();
        IOException failure =
                assertThrows(
                        IOException.class,
                        () -> Ffmpeg.run(command("sleep"), null, Duration.ofMillis(50), 1024));
        assertTrue(failure.getMessage().contains("timed out"), failure.getMessage());
        assertTrue(Duration.ofNanos(System.nanoTime() - start).toSeconds() < 3);
    }

    private static List<String> command(String mode) {
        return List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp",
                System.getProperty("java.class.path"),
                Fixture.class.getName(),
                mode);
    }

    public static final class Fixture {
        public static void main(String[] args) throws Exception {
            switch (args[0]) {
                case "small" -> System.out.print("ok");
                case "large" -> System.out.write(new byte[8192]);
                case "sleep" -> Thread.sleep(30_000);
                default -> throw new IllegalArgumentException(args[0]);
            }
        }
    }
}
