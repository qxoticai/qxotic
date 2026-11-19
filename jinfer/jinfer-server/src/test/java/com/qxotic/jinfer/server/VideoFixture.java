package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;

/**
 * A real mp4 for the tests that send video over HTTP: the six parity frames, two frames each, at 4
 * fps. ffmpeg is the sampler's own seam, and a test without ffmpeg skips.
 */
final class VideoFixture {

    private VideoFixture() {}

    /** Whether ffmpeg runs here at all. */
    static boolean available() {
        try {
            Process probe =
                    new ProcessBuilder("ffmpeg", "-version").redirectErrorStream(true).start();
            probe.getInputStream().readAllBytes();
            return probe.waitFor(30, TimeUnit.SECONDS) && probe.exitValue() == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    /**
     * The 3 s clip of {@code test-fixtures/lfm3/video}: the six frames llama.cpp samples, each
     * twice, so the server's own 2 fps sampler recovers them. The caller deletes it.
     */
    static Path clip() throws Exception {
        Path dir = Files.createTempDirectory("d1-video");
        int[] sequence = {0, 0, 2, 2, 4, 4, 7, 7, 9, 9, 11, 11};
        Path source = TestModels.fixture("lfm3/video");
        for (int i = 0; i < sequence.length; i++)
            Files.copy(
                    source.resolve(String.format("clip-%02d.png", sequence[i])),
                    dir.resolve(String.format("frame-%02d.png", i)),
                    StandardCopyOption.REPLACE_EXISTING);
        Path out = dir.resolve("clip.mp4");
        Process ffmpeg =
                new ProcessBuilder(
                                "ffmpeg",
                                "-y",
                                "-loglevel",
                                "error",
                                "-start_number",
                                "0",
                                "-framerate",
                                "4",
                                "-i",
                                "frame-%02d.png",
                                "-c:v",
                                "libx264",
                                "-pix_fmt",
                                "yuv420p",
                                "-t",
                                "3",
                                out.toString())
                        .directory(dir.toFile())
                        .redirectErrorStream(true)
                        .start();
        assertTrue(ffmpeg.waitFor(120, TimeUnit.SECONDS), "ffmpeg timed out");
        String log = new String(ffmpeg.getInputStream().readAllBytes());
        assertEquals(0, ffmpeg.exitValue(), log);
        assertTrue(Files.size(out) > 0, "empty clip");
        return out;
    }
}
