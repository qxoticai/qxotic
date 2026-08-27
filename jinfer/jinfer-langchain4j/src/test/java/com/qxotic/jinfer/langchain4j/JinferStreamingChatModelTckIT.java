package com.qxotic.jinfer.langchain4j;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.common.AbstractStreamingChatModelIT;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;
import org.mockito.InOrder;
import org.mockito.Mockito;
import dev.langchain4j.data.message.ImageContent;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The langchain4j streaming compliance kit against {@link JinferStreamingChatModel} on LFM2.5-8B.
 * Same capability map as {@link JinferChatModelTckIT}; streaming callbacks arrive on a virtual
 * thread per request. Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferStreamingChatModelTckIT#modelAvailable")
class JinferStreamingChatModelTckIT extends AbstractStreamingChatModelIT {

    static boolean modelAvailable() {
        return TestModels.find(JinferChatModelTckIT.REF).isPresent();
    }

    private static JinferChatModel shared;

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(StreamingChatModel model) {
        return JinferTokenUsage.class;
    }

    @AfterAll
    static void unloadShared() {
        if (shared != null) shared.close();
    }

    // ---- the blocking kit's per-model capability gates, same reasons ----

    @Override
    @ParameterizedTest
    @MethodSource("modelsSupportingTools")
    @EnabledIf("supportsTools")
    protected void should_execute_a_tool_then_answer(StreamingChatModel model) {
        JinferChatModelTckIT.assumeNotBareSpecMarginal();
        super.should_execute_a_tool_then_answer(model);
    }

    @Override
    @ParameterizedTest
    @MethodSource("modelsSupportingTools")
    @EnabledIf("supportsTools")
    protected void should_execute_multiple_tools_in_parallel_then_answer(StreamingChatModel model) {
        JinferChatModelTckIT.assumeParallelCallsRepresentable();
        super.should_execute_multiple_tools_in_parallel_then_answer(model);
    }

    @Override
    @ParameterizedTest
    @MethodSource("models")
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void should_respect_maxOutputTokens_in_chat_request(StreamingChatModel model) {
        JinferChatModelTckIT.assumeReasoningFitsTheBudget();
        super.should_respect_maxOutputTokens_in_chat_request(model);
    }

    @Override
    @Test
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void should_respect_maxOutputTokens_in_default_model_parameters() {
        JinferChatModelTckIT.assumeReasoningFitsTheBudget();
        super.should_respect_maxOutputTokens_in_default_model_parameters();
    }

    @Override
    @ParameterizedTest
    @MethodSource("models")
    @EnabledIf("supportsMaxOutputTokensParameter")
    protected void
            should_respect_common_parameters_wrapped_in_integration_specific_class_in_chat_request(
                    StreamingChatModel model) {
        JinferChatModelTckIT.assumeReasoningFitsTheBudget();
        super
                .should_respect_common_parameters_wrapped_in_integration_specific_class_in_chat_request(
                        model);
    }

    @Override
    protected List<StreamingChatModel> models() {
        // one shared model behind a non-AutoCloseable wrapper - see the blocking TCK's
        // models() note for why (JUnit autocloses arguments; fresh-per-call OOMs collection)
        if (shared == null) {
            shared = blocking(List.of());
        }
        StreamingChatModel m = shared.streaming();
        return List.of(
                new StreamingChatModel() {
                    @Override
                    public void chat(ChatRequest request, StreamingChatResponseHandler handler) {
                        m.chat(request, handler);
                    }

                    @Override
                    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
                        m.chat(request, handler);
                    }

                    @Override
                    public ChatRequestParameters defaultRequestParameters() {
                        return m.defaultRequestParameters();
                    }
                });
    }

    // see the blocking TCK: createModelWith products are not parameterized arguments, so
    // nothing closes them unless we track them ourselves
    private static final List<JinferChatModel> created =
            Collections.synchronizedList(new ArrayList<>());

    @AfterAll
    static void unloadCreated() {
        created.forEach(JinferChatModel::close);
        created.clear();
    }

    private static JinferChatModel track(JinferChatModel m) {
        created.add(m);
        return m;
    }

    @Override
    public StreamingChatModel createModelWith(ChatModelListener listener) {
        return track(blocking(List.of(listener))).streaming();
    }

    @Override
    protected StreamingChatModel createModelWith(ChatRequestParameters parameters) {
        var builder =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                        .contextLength(8192)
                        .defaultRequestParameters(parameters)
                        // same greedy pinning as models(); kit parameters override
                        .temperature(0.0)
                        .thinking(JinferChatModelTckIT.tckThinking())
                        .seed(7L);
        if (JinferChatModelTckIT.mediaAvailable()) {
            builder.companionPath("media", Path.of(JinferChatModelTckIT.MEDIA));
        }
        return track(builder.build()).streaming();
    }

    @Override
    protected ChatRequestParameters createIntegrationSpecificParameters(int maxOutputTokens) {
        return DefaultChatRequestParameters.builder().maxOutputTokens(maxOutputTokens).build();
    }

    private static JinferChatModel blocking(List<ChatModelListener> listeners) {
        var builder =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                        .contextLength(8192)
                        .maxOutputTokens(512)
                        // pinned GREEDY like the blocking TCK's models(): a compliance suite must
                        // not flake, and a temperature draw at a near-tie flips with cache-state
                        // drift
                        .temperature(0.0)
                        .thinking(JinferChatModelTckIT.tckThinking())
                        .seed(7L)
                        .listeners(listeners);
        if (JinferChatModelTckIT.mediaAvailable()) {
            builder.companionPath("media", Path.of(JinferChatModelTckIT.MEDIA));
        }
        return builder.build();
    }

    // ---- same capability map as the blocking TCK ----

    @Override
    protected boolean supportsPartialToolStreaming(StreamingChatModel model) {
        return false; // tool calls are announced complete (onCompleteToolCall), not
        // argument-streamed
    }

    // ---- exact callback sequence: thinking* then content* then one CompleteToolCall each ----

    @Override
    protected void verifyToolCallbacks(
            StreamingChatResponseHandler handler, InOrder io, String id) {
        io.verify(handler, Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, Mockito.atLeast(0)).onPartialResponse(any(), any());
        io.verify(handler)
                .onCompleteToolCall(
                        argThat(
                                call ->
                                        call.index() == 0
                                                && call.toolExecutionRequest()
                                                        .name()
                                                        .equals("getWeather")));
    }

    @Override
    protected void verifyToolCallbacks(
            StreamingChatResponseHandler handler, InOrder io, StreamingChatModel model) {
        io.verify(handler, Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, Mockito.atLeast(0)).onPartialResponse(any(), any());
        io.verify(handler).onCompleteToolCall(any());
    }

    @Override
    protected void verifyToolCallbacks(
            StreamingChatResponseHandler handler, InOrder io, String id1, String id2) {
        io.verify(handler, Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, Mockito.atLeast(0)).onPartialResponse(any(), any());
        io.verify(handler)
                .onCompleteToolCall(
                        argThat(
                                call ->
                                        call.index() == 0
                                                && call.toolExecutionRequest()
                                                        .name()
                                                        .equals("getWeather")));
        io.verify(handler)
                .onCompleteToolCall(
                        argThat(
                                call ->
                                        call.index() == 1
                                                && call.toolExecutionRequest()
                                                        .name()
                                                        .equals("getTime")));
    }

    @Override
    protected boolean supportsModelNameParameter() {
        return false;
    }

    @Override
    protected boolean supportsToolsAndJsonResponseFormatWithSchema() {
        return true; // the schema rides the family reply language; calls stay the family's own
    }

    @Override
    protected boolean supportsSingleImageInputAsBase64EncodedString() {
        return JinferChatModelTckIT.mediaAvailable();
    }

    @Override
    protected boolean supportsMultipleImageInputsAsBase64EncodedStrings() {
        return JinferChatModelTckIT.mediaAvailable();
    }

    @Override
    protected boolean supportsSingleImageInputAsPublicURL() {
        return false;
    }

    // the kit's photos vendored locally - see the blocking TCK's note
    @Override
    protected ImageContent catImageContentBase64() {
        return ImageContent.from(
                JinferChatModelTckIT.kitImage("cat.png"), "image/png");
    }

    @Override
    protected ImageContent diceImageContentBase64() {
        return ImageContent.from(
                JinferChatModelTckIT.kitImage("dice.png"), "image/png");
    }
}
