package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.common.AbstractChatModelIT;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.TokenUsage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The langchain4j compliance kit (AbstractChatModelIT: "all the common tests that every ChatModel
 * must successfully pass") against JinferChatModel on LFM2.5-8B. Capability switches document
 * exactly what this provider rejects by design (stop strings, JSON formats, toolChoice REQUIRED,
 * per-request model switching, image URLs). Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferChatModelTckIT#modelAvailable")
class JinferChatModelTckIT extends AbstractChatModelIT {

    static final String REF = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";
    static final String MEDIA = System.getProperty("jinfer.testMedia", "");

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    /**
     * The kit's models run thinking-ON except SmolLM3: the 3B force-closed mid-thought by the
     * reasoning cap fabricates a turn header next (its recorded habit), which is a stop since the
     * turn-guard fix - every behavioral test then sees a whitespace-only answer. Thinking off is a
     * supported first-class mode for this family and answers every kit shape correctly; the
     * thinking-ON path keeps its coverage from the other seven families.
     */
    static boolean tckThinking() {
        return !TestModels.require(REF).toString().contains("SmolLM3");
    }

    private static JinferChatModel model;

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(ChatModel model) {
        return JinferTokenUsage.class;
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
            var builder =
                    JinferChatModel.builder()
                            .modelPath(TestModels.require(REF))
                            .contextLength(8192)
                            .maxOutputTokens(512) // bound unconstrained TCK requests
                            // pinned GREEDY: a compliance suite must not flake. A seed alone is
                            // not enough - block-cache state drifts an ulp across suite orders
                            // and a temperature draw at a near-tie flips with it (observed: a
                            // tool round answering with no text). The kit tests the CONTRACT,
                            // not sampling quality.
                            .temperature(0.0)
                            .thinking(tckThinking())
                            .seed(7L);
            if (mediaAvailable()) builder.companionPath("media", Path.of(MEDIA));
            model = builder.build();
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

    /**
     * The kit's weather round uses a BARE tool spec (empty description, nothing required); on the
     * smallest checkpoints the round-2 continuation after the tool result is a first-token near-tie
     * between answering and closing the turn, flipped by suite-order cache drift and JIT-warmup
     * jitter (isolated runs of the same request answer correctly). The wire is byte-faithful
     * (gemma's verified against the checkpoint's own embedded Jinja); richer specs answer reliably
     * - capability, not the provider, and the family batteries gate the same scenario ({@code
     * Gemma4ToolIT}, {@code SmolLm3ToolIT}). Every other model stays strict.
     */
    static void assumeNotBareSpecMarginal() {
        String model = TestModels.require(REF).toString();
        Assumptions.assumeFalse(
                model.contains("gemma-4-E2B") || model.contains("SmolLM3"),
                "small-checkpoint bare-spec round-2 near-tie (capability; see the family battery)");
    }

    /**
     * gpt-oss ALWAYS reasons - Harmony has no think markers to disable ({@code thinking} is a
     * documented no-op; the knob is the preamble's {@code Reasoning:} line) - so the kit's 5-token
     * budgets die inside the analysis channel before any {@code final} text exists, exactly like a
     * hosted reasoning model returning empty content under a tiny completion cap. Usage and the
     * LENGTH finish stay asserted by the runs that pass; only "text not blank" cannot hold.
     */
    static void assumeReasoningFitsTheBudget() {
        Assumptions.assumeFalse(
                TestModels.require(REF).toString().contains("gpt-oss"),
                "gpt-oss: a 5-token budget cannot surface final-channel text on an"
                        + " always-reasoning family");
    }

    /**
     * Harmony's template renders at most ONE call per assistant message (extra calls are
     * unrenderable in the family's own echo), so gpt-oss answering the parallel-calls prompt with
     * one call is correct family behavior, not a miss.
     */
    static void assumeParallelCallsRepresentable() {
        Assumptions.assumeFalse(
                TestModels.require(REF).toString().contains("gpt-oss"),
                "gpt-oss: Harmony renders at most one call per assistant message");
    }

    @Override
    @ParameterizedTest
    @MethodSource("modelsSupportingTools")
    @EnabledIf("supportsTools")
    protected void should_execute_a_tool_then_answer(ChatModel model) {
        assumeNotBareSpecMarginal();
        super.should_execute_a_tool_then_answer(model);
    }

    @Override
    @ParameterizedTest
    @MethodSource("modelsSupportingTools")
    @EnabledIf("supportsTools")
    protected void should_execute_multiple_tools_in_parallel_then_answer(ChatModel model) {
        assumeParallelCallsRepresentable();
        super.should_execute_multiple_tools_in_parallel_then_answer(model);
    }

    @Override
    @ParameterizedTest
    @MethodSource("models")
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void should_respect_maxOutputTokens_in_chat_request(ChatModel model) {
        assumeReasoningFitsTheBudget();
        super.should_respect_maxOutputTokens_in_chat_request(model);
    }

    @Override
    @Test
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void should_respect_maxOutputTokens_in_default_model_parameters() {
        assumeReasoningFitsTheBudget();
        super.should_respect_maxOutputTokens_in_default_model_parameters();
    }

    @Override
    @ParameterizedTest
    @MethodSource("models")
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void
            should_respect_common_parameters_wrapped_in_integration_specific_class_in_chat_request(
                    ChatModel model) {
        assumeReasoningFitsTheBudget();
        super
                .should_respect_common_parameters_wrapped_in_integration_specific_class_in_chat_request(
                        model);
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
        var builder =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(8192)
                        .defaultRequestParameters(parameters)
                        // same reason as models(); the kit's own parameters override where set
                        .temperature(0.0)
                        .thinking(tckThinking())
                        .seed(7L);
        if (mediaAvailable()) builder.companionPath("media", Path.of(MEDIA));
        JinferChatModel m = builder.build();
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
        return true; // the schema rides the family reply language; calls stay the family's own
    }

    @Override
    protected boolean supportsSingleImageInputAsBase64EncodedString() {
        return mediaAvailable();
    }

    @Override
    protected boolean supportsMultipleImageInputsAsBase64EncodedStrings() {
        return mediaAvailable();
    }

    @Override
    protected boolean supportsSingleImageInputAsPublicURL() {
        return false; // this library never fetches over the network
    }

    /**
     * The kit's two photos, vendored as test resources: its defaults pull them from wikimedia over
     * the network per invocation, which would make a COMPLIANCE suite depend on uptime. The bytes
     * are byte-identical to the kit's URLs (the assertion vocabulary - "cat", "dice" - is semantic,
     * so the images must be the real ones).
     */
    @Override
    protected ImageContent catImageContentBase64() {
        return ImageContent.from(kitImage("cat.png"), "image/png");
    }

    @Override
    protected ImageContent diceImageContentBase64() {
        return ImageContent.from(kitImage("dice.png"), "image/png");
    }

    static String kitImage(String name) {
        try (var in = JinferChatModelTckIT.class.getResourceAsStream("/kit-images/" + name)) {
            if (in == null) throw new IllegalStateException("missing test resource " + name);
            return Base64.getEncoder().encodeToString(in.readAllBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static boolean mediaAvailable() {
        return !MEDIA.isBlank() && Files.isRegularFile(Path.of(MEDIA));
    }
}
