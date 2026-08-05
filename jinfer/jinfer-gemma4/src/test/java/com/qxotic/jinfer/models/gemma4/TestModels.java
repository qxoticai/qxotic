package com.qxotic.jinfer.models.gemma4;

import java.nio.file.Path;

/** Local model fixtures shared by the gemma4 parity tests; tests skip when absent. */
final class TestModels {
    static final Path E2B_MMPROJ =
            Path.of(
                    System.getProperty(
                            "jinfer.test.e2bMmproj",
                            "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"));
    static final Path B12_MMPROJ =
            Path.of(
                    System.getProperty(
                            "jinfer.test.12bMmproj",
                            "/home/mukel/Desktop/playground/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf"));

    private TestModels() {}
}
