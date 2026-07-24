package com.qxotic.jinfer.langchain4j;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.common.AbstractStreamingChatModelIT;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j streaming compliance kit against {@link JinferStreamingChatModel} on LFM2.5-8B.
 * Same capability map as {@link JinferChatModelTckIT}; streaming callbacks arrive on a virtual
 * thread per request. Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferStreamingChatModelTckIT#modelAvailable")
class JinferStreamingChatModelTckIT extends AbstractStreamingChatModelIT {

    static boolean modelAvailable() {
        return Files.exists(JinferChatModelTckIT.MODEL);
    }

    private static StreamingChatModel model;

    @Override
    protected List<StreamingChatModel> models() {
        if (model == null) {
            model = blocking(List.of()).streaming();
        }
        return List.of(model);
    }

    @Override
    public StreamingChatModel createModelWith(ChatModelListener listener) {
        return blocking(List.of(listener)).streaming();
    }

    @Override
    protected StreamingChatModel createModelWith(
            dev.langchain4j.model.chat.request.ChatRequestParameters parameters) {
        return JinferChatModel.builder()
                .modelPath(JinferChatModelTckIT.MODEL)
                .contextLength(8192)
                .defaultRequestParameters(parameters)
                .build()
                .streaming();
    }

    @Override
    protected dev.langchain4j.model.chat.request.ChatRequestParameters
            createIntegrationSpecificParameters(int maxOutputTokens) {
        return dev.langchain4j.model.chat.request.DefaultChatRequestParameters.builder()
                .maxOutputTokens(maxOutputTokens)
                .build();
    }

    private static JinferChatModel blocking(List<ChatModelListener> listeners) {
        return JinferChatModel.builder()
                .modelPath(JinferChatModelTckIT.MODEL)
                .contextLength(8192)
                .maxOutputTokens(512)
                .listeners(listeners)
                .build();
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
            dev.langchain4j.model.chat.response.StreamingChatResponseHandler handler,
            org.mockito.InOrder io,
            String id) {
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialResponse(any(), any());
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
            dev.langchain4j.model.chat.response.StreamingChatResponseHandler handler,
            org.mockito.InOrder io,
            StreamingChatModel model) {
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialResponse(any(), any());
        io.verify(handler).onCompleteToolCall(any());
    }

    @Override
    protected void verifyToolCallbacks(
            dev.langchain4j.model.chat.response.StreamingChatResponseHandler handler,
            org.mockito.InOrder io,
            String id1,
            String id2) {
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialThinking(any());
        io.verify(handler, org.mockito.Mockito.atLeast(0)).onPartialResponse(any(), any());
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
    protected boolean supportsStopSequencesParameter() {
        return false;
    }

    @Override
    protected boolean supportsToolsAndJsonResponseFormatWithSchema() {
        return false;
    }

    @Override
    protected boolean supportsSingleImageInputAsBase64EncodedString() {
        return false;
    }

    @Override
    protected boolean supportsSingleImageInputAsPublicURL() {
        return false;
    }

    @Override
    protected boolean assertResponseId() {
        return false;
    }

    @Override
    protected boolean assertToolId(StreamingChatModel model) {
        return false;
    }

    @Override
    protected boolean assertExceptionType() {
        return false;
    }
}
