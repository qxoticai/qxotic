// frozen single-prompt (use case A) validation on LFM2.5 via the shared testkit scenario.
//   java ... com.qxotic.jinfer.models.lfm2.Lfm2FrozenPromptRun [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.FrozenPromptScenario;
import com.qxotic.jinfer.testkit.Harness;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Lfm2FrozenPromptRun {
    @Test
    @Tag("driver")
    void run() throws Exception {
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        Path path =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        Lfm2 m = Lfm2.loadModel(path, 8192);
        new FrozenPromptScenario<>(
                        new Harness<>(m.loaded(), m.turnTemplate().orElseThrow(), path, 8192))
                .run("Lfm2FrozenPromptRun");
    }
}
