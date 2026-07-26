package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code AutoCloseable} semantics against the small LFM2 (cheap to load per test) - the mirror of
 * the spring twin's lifecycle contract. Model-gated: assume-skips when the file is absent.
 */
@Tag("integration")
class JinferLifecycleIT {

    static final Path SMALL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelSmall", ModelFixture.LFM25_350M_Q8.path().toString()));

    private static JinferChatModel load() {
        return JinferChatModel.builder()
                .modelPath(SMALL)
                .contextLength(2048)
                .maxOutputTokens(8)
                .build();
    }

    @Test
    void closeGuardsEveryEntryPointAndIsIdempotent() throws Exception {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        m.chat(UserMessage.from("hi")); // proves the model worked before close
        m.close();
        m.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> m.chat(UserMessage.from("hi")));
        assertThrows(
                IllegalStateException.class,
                () ->
                        m.streaming()
                                .chat(
                                        "hi",
                                        new dev.langchain4j.model.chat.response
                                                .StreamingChatResponseHandler() {
                                            @Override
                                            public void onPartialResponse(String partial) {}

                                            @Override
                                            public void onCompleteResponse(
                                                    dev.langchain4j.model.chat.response.ChatResponse
                                                            response) {}

                                            @Override
                                            public void onError(Throwable error) {}
                                        }));
        assertThrows(
                IllegalStateException.class,
                () -> m.withCachedPrompt(List.of(SystemMessage.from("x")), List.of()));
        assertThrows(
                IllegalStateException.class,
                () -> m.saveCachedPrompts(Path.of("/tmp/jinfer-closed.jkv")));
    }

    @Test
    void closingTheBaseClosesViews() {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        JinferChatModel view =
                m.withCachedPrompt(List.of(SystemMessage.from("You are terse.")), List.of());
        m.close();
        assertThrows(IllegalStateException.class, () -> view.chat(UserMessage.from("hi")));
    }
}
