// PromptCache validation + benchmark on Granite 4.1 3B (uniform full attention - the degenerate
// codec) via the shared testkit scenario.   java ... com.qxotic.jinfer.models.llama.GraniteCacheRun
// [model.gguf]
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.testkit.CacheScenario;
import com.qxotic.jinfer.testkit.Harness;
import com.qxotic.jinfer.testkit.Stories;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class GraniteCacheRun {
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
                                : "/home/mukel/Desktop/playground/models/ibm-granite/granite-4.1-3b-Q8_0.gguf");
        Granite m = Granite.loadModel(path, 8192);
        Harness<Granite.State> h =
                new Harness<>(m.loaded(), m.turnTemplate().orElseThrow(), path, 8192);
        new CacheScenario<>(
                        h,
                        CacheScenario.Config.of(
                                "You are a concise assistant.",
                                60,
                                new CacheScenario.LongCase(
                                        Stories.expeditionLog(),
                                        "How many entries were there? One number.",
                                        1500)))
                .run("GraniteCacheRun");
    }
}
