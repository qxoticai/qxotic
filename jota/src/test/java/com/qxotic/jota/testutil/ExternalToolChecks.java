package com.qxotic.jota.testutil;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public final class ExternalToolChecks {

    private static final long TIMEOUT_SECONDS =
            Long.getLong("jota.test.external.tool.timeout.seconds", 10L);
    private static final Map<String, Boolean> VERSION_CACHE = new ConcurrentHashMap<>();

    private ExternalToolChecks() {}

    public static boolean hasVersionCommand(String executable) {
        return VERSION_CACHE.computeIfAbsent(executable, ExternalToolChecks::probeVersionCommand);
    }

    private static boolean probeVersionCommand(String executable) {
        ProcessBuilder builder =
                new ProcessBuilder(executable, "--version")
                        .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                        .redirectError(ProcessBuilder.Redirect.DISCARD);
        try {
            Process process = builder.start();
            if (!process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(1, TimeUnit.SECONDS);
                return false;
            }
            return process.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }
}
