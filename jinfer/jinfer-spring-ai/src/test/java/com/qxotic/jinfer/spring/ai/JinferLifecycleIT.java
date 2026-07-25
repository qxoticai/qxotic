package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * {@code AutoCloseable} semantics against the small LFM2 (cheap to load per test). Model-gated:
 * assume-skips when the file is absent. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class JinferLifecycleIT {

    static final Path SMALL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelSmall", ModelFixture.LFM25_350M_Q8.path().toString()));

    private static JinferChatModel load() {
        return JinferChatModel.builder().modelPath(SMALL).contextLength(2048).maxTokens(8).build();
    }

    @Test
    void closeGuardsEveryEntryPointAndIsIdempotent() {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        m.call(new Prompt(new UserMessage("hi"))); // proves the model worked before close
        m.close();
        m.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> m.call(new Prompt(new UserMessage("hi"))));
        assertThrows(
                IllegalStateException.class,
                () -> m.stream(new Prompt(new UserMessage("hi"))).blockLast(Duration.ofMinutes(1)));
        assertThrows(
                IllegalStateException.class,
                () -> m.withCachedPrompt(List.of(new SystemMessage("x")), List.of()));
        assertThrows(
                IllegalStateException.class,
                () -> m.saveCachedPrompts(Path.of("/tmp/opencode/x.jkv")));
    }

    @Test
    void closingTheBaseClosesViews() {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        JinferChatModel view =
                m.withCachedPrompt(List.of(new SystemMessage("You are terse.")), List.of());
        m.close();
        assertThrows(
                IllegalStateException.class, () -> view.call(new Prompt(new UserMessage("hi"))));
    }
}
