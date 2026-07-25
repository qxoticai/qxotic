package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.common.AbstractChatModelIT;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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

    private static ChatModel model;

    @Override
    protected List<ChatModel> models() {
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(MODEL)
                            .contextLength(8192)
                            .maxOutputTokens(512) // bound unconstrained TCK requests
                            .build();
        }
        return List.of(model);
    }

    @Override
    protected ChatModel createModelWith(
            dev.langchain4j.model.chat.request.ChatRequestParameters parameters) {
        return JinferChatModel.builder()
                .modelPath(MODEL)
                .contextLength(8192)
                .defaultRequestParameters(parameters)
                .build();
    }

    @Override
    protected dev.langchain4j.model.chat.request.ChatRequestParameters
            createIntegrationSpecificParameters(int maxOutputTokens) {
        // no provider-specific parameters class: the plain defaults are the integration's shape
        return dev.langchain4j.model.chat.request.DefaultChatRequestParameters.builder()
                .maxOutputTokens(maxOutputTokens)
                .build();
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
