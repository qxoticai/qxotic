package com.qxotic.jinfer.x.boundary.media;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * The one ffmpeg process runner: streams any {@code stdin} on a daemon thread and drains stderr
 * concurrently so the child never blocks on a full pipe, returns the raw stdout bytes, and reports
 * a non-zero exit with the captured stderr. Shared by the audio, image and video codecs.
 */
final class Ffmpeg {

    private Ffmpeg() {}

    static byte[] run(List<String> cmd, byte[] stdin) throws IOException {
        Process p;
        try {
            p = new ProcessBuilder(cmd).start();
        } catch (IOException e) {
            throw new IOException("failed to launch ffmpeg (is it on PATH?): " + e.getMessage(), e);
        }
        if (stdin != null) {
            Thread feeder =
                    new Thread(
                            () -> {
                                try (OutputStream os = p.getOutputStream()) {
                                    os.write(stdin);
                                } catch (IOException ignored) {
                                    // broken pipe if ffmpeg rejects the input and exits early; the
                                    // exit code reports it
                                }
                            },
                            "ffmpeg-stdin");
            feeder.setDaemon(true);
            feeder.start();
        } else {
            p.getOutputStream().close();
        }
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        Thread errDrain =
                new Thread(
                        () -> {
                            try (InputStream es = p.getErrorStream()) {
                                es.transferTo(err);
                            } catch (IOException ignored) {
                            }
                        },
                        "ffmpeg-stderr");
        errDrain.setDaemon(true);
        errDrain.start();

        byte[] out;
        try (InputStream is = p.getInputStream()) {
            out = is.readAllBytes();
        }
        int code;
        try {
            code = p.waitFor();
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted waiting for ffmpeg", e);
        }
        try {
            errDrain.join(1000);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        if (code != 0) {
            throw new IOException(
                    "ffmpeg exited " + code + ": " + err.toString(StandardCharsets.UTF_8).strip());
        }
        return out;
    }
}
