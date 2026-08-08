package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.common.AbstractChatModelIT;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j compliance kit (AbstractChatModelIT: "all the common tests that every ChatModel
 * must successfully pass") against JinferChatModel on LFM2.5-8B. Capability switches document
 * exactly what this provider rejects by design (stop strings, JSON formats, toolChoice REQUIRED,
 * per-request model switching, image URLs). Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferChatModelTckIT#modelAvailable")
class JinferChatModelTckIT extends AbstractChatModelIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel", ModelFixture.LFM25_8B_Q8.path().toString()));

    static boolean modelAvailable() {
        return Files.exists(MODEL);
    }

    private static JinferChatModel model;

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Override
    protected List<ChatModel> models() {
        // ONE shared model, shielded: JUnit closes AutoCloseable parameterized-test arguments
        // after EVERY invocation (params autoCloseArguments default), which would kill a shared
        // model after its first test - and a fresh model per call is untenable (models() runs
        // once per test method at collection time; ~37 eager 8B loads OOM'd the fork). The
        // wrapper is deliberately NOT AutoCloseable, so JUnit has nothing to close; @AfterAll
        // closes the real model once.
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(MODEL)
                            .contextLength(8192)
                            .maxOutputTokens(512) // bound unconstrained TCK requests
                            .build();
        }
        JinferChatModel m = model;
        return List.of(
                new ChatModel() {
                    @Override
                    public ChatResponse chat(ChatRequest request) {
                        return m.chat(request);
                    }

                    @Override
                    public ChatResponse doChat(ChatRequest request) {
                        return m.chat(request);
                    }

                    @Override
                    public ChatRequestParameters defaultRequestParameters() {
                        return m.defaultRequestParameters();
                    }
                });
    }

    // createModelWith products are built INSIDE test bodies - they are not parameterized
    // arguments, so JUnit's autoCloseArguments never touches them; without tracking, each one's
    // states wait for a GC that never comes (the 30+GB TCK fork ballooning)
    private static final List<JinferChatModel> created =
            Collections.synchronizedList(new ArrayList<>());

    @AfterAll
    static void unloadCreated() {
        created.forEach(JinferChatModel::close);
        created.clear();
    }

    @Override
    protected ChatModel createModelWith(ChatRequestParameters parameters) {
        JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(8192)
                        .defaultRequestParameters(parameters)
                        .build();
        created.add(m);
        return m;
    }

    @Override
    protected ChatRequestParameters createIntegrationSpecificParameters(int maxOutputTokens) {
        // no provider-specific parameters class: the plain defaults are the integration's shape
        return DefaultChatRequestParameters.builder().maxOutputTokens(maxOutputTokens).build();
    }

    // ---- capabilities this provider rejects by design ----

    @Override
    protected boolean supportsModelNameParameter() {
        return false; // one loaded GGUF per model instance; no per-request switching
    }

    @Override
    protected boolean supportsToolsAndJsonResponseFormatWithSchema() {
        return false;
    }

    @Override
    protected boolean supportsSingleImageInputAsBase64EncodedString() {
        return false; // LFM2 is text-only (gemma4 does media; it lacks tool support for the TCK)
    }

    @Override
    protected boolean supportsSingleImageInputAsPublicURL() {
        return false; // this library never fetches over the network
    }
}
