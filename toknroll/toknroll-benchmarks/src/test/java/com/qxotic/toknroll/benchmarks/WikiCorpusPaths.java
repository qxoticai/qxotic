package com.qxotic.toknroll.benchmarks;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class WikiCorpusPaths {

    private static final String HOME = System.getProperty("user.home");

    private WikiCorpusPaths() {}

    static Path enwik8() {
        return resolve(
                "enwik8",
                "toknroll.enwik8.path",
                Path.of(HOME, ".cache", "qxotic", "tokenizers", "corpus", "enwik8"));
    }

    static Path enwik9() {
        return resolve(
                "enwik9",
                "toknroll.enwik9.path",
                Path.of(HOME, ".cache", "qxotic", "tokenizers", "corpus", "enwik9"));
    }

    static Path forCorpus(String corpus) {
        if ("enwik8".equals(corpus)) {
            return enwik8();
        }
        if ("enwik9".equals(corpus)) {
            return enwik9();
        }
        throw new IllegalArgumentException("Unsupported corpus: " + corpus);
    }

    private static Path resolve(String name, String property, Path... fallbacks) {
        List<Path> checked = new ArrayList<Path>();
        String configured = System.getProperty(property);
        if (configured != null && !configured.isBlank()) {
            Path configuredPath = Path.of(configured);
            checked.add(configuredPath);
            if (Files.exists(configuredPath)) {
                return configuredPath;
            }
        }
        for (Path candidate : fallbacks) {
            checked.add(candidate);
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        throw new IllegalStateException(
                "Could not locate "
                        + name
                        + ". Set -D"
                        + property
                        + "=/path/to/"
                        + name
                        + " or run toknroll-benchmarks/run_enwik_tests.py to download cache. Checked: "
                        + checked);
    }
}
