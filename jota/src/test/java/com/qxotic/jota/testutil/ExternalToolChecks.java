package com.qxotic.jota.testutil;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public final class ExternalToolChecks {

    private static final long TIMEOUT_SECONDS =
            Long.getLong("jota.test.external.tool.timeout.seconds", 10L);
    private static final Map<String, Boolean> VERSION_CACHE = new ConcurrentHashMap<>();
    private static final Map<List<String>, Boolean> COMMAND_CACHE = new ConcurrentHashMap<>();

    private ExternalToolChecks() {}

    public static boolean hasVersionCommand(String executable) {
        return VERSION_CACHE.computeIfAbsent(executable, ExternalToolChecks::probeVersionCommand);
    }

    public static boolean hasCommand(String... command) {
        if (command == null || command.length == 0) {
            return false;
        }
        return COMMAND_CACHE.computeIfAbsent(
                List.copyOf(Arrays.asList(command)), ExternalToolChecks::probeCommand);
    }

    private static boolean probeVersionCommand(String executable) {
        return probeCommand(List.of(executable, "--version"));
    }

    private static boolean probeCommand(List<String> command) {
        ProcessBuilder builder =
                new ProcessBuilder(command)
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
