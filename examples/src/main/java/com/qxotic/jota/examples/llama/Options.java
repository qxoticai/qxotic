package com.qxotic.jota.examples.llama;

import com.qxotic.jota.Device;
import java.nio.file.Path;

final class Options {
    final Path modelPath;
    final String prompt;
    final String systemPrompt;
    final boolean interactive;
    final float temperature;
    final float topP;
    final long seed;
    final int maxTokens;
    final int benchmarkRuns;
    final long benchmarkPauseMs;
    final boolean stream;
    final boolean trace;
    final Device device;

    private Options(
            Path modelPath,
            String prompt,
            String systemPrompt,
            boolean interactive,
            float temperature,
            float topP,
            long seed,
            int maxTokens,
            int benchmarkRuns,
            long benchmarkPauseMs,
            boolean stream,
            boolean trace,
            Device device) {
        this.modelPath = modelPath;
        this.prompt = prompt;
        this.systemPrompt = systemPrompt;
        this.interactive = interactive;
        this.temperature = temperature;
        this.topP = topP;
        this.seed = seed;
        this.maxTokens = maxTokens;
        this.benchmarkRuns = benchmarkRuns;
        this.benchmarkPauseMs = benchmarkPauseMs;
        this.stream = stream;
        this.trace = trace;
        this.device = device;
    }

    static Options parse(String[] args) {
        Path modelPath = null;
        String prompt = null;
        String systemPrompt = null;
        boolean interactive = false;
        float temperature = 0.1f;
        float topP = 0.95f;
        long seed = System.nanoTime();
        int maxTokens = 512;
        int benchmarkRuns = 1;
        long benchmarkPauseMs = 0L;
        boolean stream = true;
        boolean trace = Boolean.getBoolean("com.qxotic.trace");
        Device device = Device.PANAMA;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            String value;
            if (arg.contains("=")) {
                String[] parts = arg.split("=", 2);
                arg = parts[0];
                value = parts[1];
            } else {
                value = i + 1 < args.length ? args[i + 1] : null;
            }
            switch (arg) {
                case "--model", "-m" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    modelPath = Path.of(value);
                }
                case "--prompt", "-p" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    prompt = value;
                }
                case "--system-prompt", "-sp" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    systemPrompt = value;
                }
                case "--interactive", "--chat", "-i" -> interactive = true;
                case "--temperature", "--temp" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    temperature = Float.parseFloat(value);
                }
                case "--top-p" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    topP = Float.parseFloat(value);
                }
                case "--seed", "-s" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    seed = Long.parseLong(value);
                }
                case "--max-tokens", "-n" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    maxTokens = Integer.parseInt(value);
                }
                case "--benchmark-runs", "--bench-runs" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    benchmarkRuns = Integer.parseInt(value);
                }
                case "--benchmark-pause-ms", "--bench-pause-ms" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    benchmarkPauseMs = Long.parseLong(value);
                }
                case "--stream" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    stream = Boolean.parseBoolean(value);
                }
                case "--trace" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    trace = Boolean.parseBoolean(value);
                }
                case "--device", "-d" -> {
                    if (!arg.contains("=")) {
                        i++;
                    }
                    device = parseDevice(value);
                }
                case "--help", "-h" -> {
                    printUsage();
                    System.exit(0);
                }
                default -> throw new IllegalArgumentException("Unknown option: " + arg);
            }
        }
        if (modelPath == null) {
            throw new IllegalArgumentException("--model is required");
        }
        if (!interactive && (prompt == null || prompt.isBlank())) {
            throw new IllegalArgumentException(
                    "--prompt is required unless --interactive is enabled");
        }
        return new Options(
                modelPath,
                prompt,
                systemPrompt,
                interactive,
                temperature,
                topP,
                seed,
                maxTokens,
                benchmarkRuns,
                benchmarkPauseMs,
                stream,
                trace,
                device);
    }

    private static Device parseDevice(String value) {
        String normalized = value == null ? "panama" : value.trim().toLowerCase();
        return switch (normalized) {
            case "panama", "cpu", "native" -> Device.PANAMA;
            case "c" -> Device.C;
            case "hip" -> Device.HIP;
            case "opencl", "ocl" -> Device.OPENCL;
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported --device value: " + value + " (use panama|c|hip|opencl)");
        };
    }

    private static void printUsage() {
        System.out.println("Usage: Llama32Cli --model <path> [options]");
    }
}
