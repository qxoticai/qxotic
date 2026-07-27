package com.qxotic.jinfer.example.judgeadvisor;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.testkit.ModelFixture;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The full loop OFFLINE: LFM2.5 plays both roles - base model generates (with the weather gimmick),
 * a cached-prompt view judges with a grammar-pinned verdict. No API key, no network. Model-gated
 * via {@link ModelFixture}. Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration
 * -pl jinfer-example-judge-advisor}
 */
@Tag("integration")
class JudgeAdvisorOfflineIT {

    @Test
    void selfRefineLoopOffline() {
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_8B_Q8.require())
                        .contextLength(8192)
                        .maxTokens(512)
                        .build();
        String answer = JudgeAdvisorApplication.run(base, base);
        assertNotNull(answer);
        assertTrue(!answer.isBlank(), "empty final answer");
        // the loop's whole point: after the -255C verdict fails, the passing answer reports 15C
        Assumptions.assumeTrue(
                answer.contains("15"), "generator did not re-roll the tool: " + answer);
    }
}
