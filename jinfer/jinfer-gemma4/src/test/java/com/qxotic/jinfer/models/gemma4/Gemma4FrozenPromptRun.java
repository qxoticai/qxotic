// frozen single-prompt (use case A) validation on Gemma 4 E2B via the shared testkit scenario - the
// sealed span includes the SWA ring checkpoints, proving the single-span format handles windowed
// layers.   java ... com.qxotic.jinfer.models.gemma4.Gemma4FrozenPromptRun [model.gguf]
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.FrozenPromptScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Gemma4FrozenPromptRun {
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
                Path.of(args.length > 0 ? args[0] : ModelFixture.GEMMA4_E2B_Q8.path().toString());
        Gemma4 m = Gemma4.loadModel(path, Arena.ofAuto());
        new FrozenPromptScenario<>(
                        new Harness<>(m.loaded(), m.turnTemplate().orElseThrow(), path, 8192))
                .run("Gemma4FrozenPromptRun");
    }
}
