package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.common.AbstractStreamingChatModelListenerIT;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j streaming listener compliance kit against {@link JinferStreamingChatModel}. The
 * request/response half runs as shipped; the error half is REPLACED: core's {@code
 * StreamingChatModel.chat} calls {@code doChat} with no try/catch, so a request jinfer rejects
 * synchronously (here: a closed engine) reaches the caller as a thrown exception and never reaches
 * a listener - the divergence documented on {@link JinferStreamingChatModel}. This pins that exact
 * behavior so an upstream core fix surfaces here as a failing test, not a surprise.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferChatModelListenerIT#modelAvailable")
class JinferStreamingChatModelListenerIT extends AbstractStreamingChatModelListenerIT {

    // kit products are built inside test bodies and nothing closes them - track, like the TCK
    private static final List<JinferChatModel> created =
            Collections.synchronizedList(new ArrayList<>());

    @AfterAll
    static void unload() {
        created.forEach(JinferChatModel::close);
        created.clear();
    }

    private String name;

    @Override
    protected StreamingChatModel createModel(ChatModelListener listener) {
        JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelListenerIT.REF))
                        .contextLength(4096)
                        // the kit's defaults, explicit: thinking OFF so the 7-token budget buys
                        // answer tokens, not analysis
                        .temperature(0.7)
                        .topP(1.0)
                        .maxOutputTokens(7)
                        .thinking(false)
                        .listeners(List.of(listener))
                        .build();
        name = m.defaultRequestParameters().modelName();
        created.add(m);
        return m.streaming();
    }

    @Override
    protected String modelName() {
        return name; // captured from the live engine: the kit compares against what WE report
    }

    @Override
    protected StreamingChatModel createFailingModel(ChatModelListener listener) {
        JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelListenerIT.REF))
                        .contextLength(512)
                        .listeners(List.of(listener))
                        .build();
        m.close(); // use-after-close: jinfer's honest call-time failure
        return m.streaming();
    }

    @Override
    protected Class<? extends Exception> expectedExceptionClass() {
        return IllegalStateException.class;
    }

    @Override
    @Test
    protected void should_listen_error() throws Exception {
        // the kit's version waits on onError, which a SYNCHRONOUS rejection never produces here.
        // jinfer's contract: the failure throws raw out of chat(); the listener saw the request,
        // and exactly nothing after it (core reports no terminal event on the streaming path)
        AtomicInteger requests = new AtomicInteger();
        AtomicInteger terminal = new AtomicInteger();
        ChatModelListener counting =
                new ChatModelListener() {
                    @Override
                    public void onRequest(
                            dev.langchain4j.model.chat.listener.ChatModelRequestContext context) {
                        requests.incrementAndGet();
                    }

                    @Override
                    public void onResponse(
                            dev.langchain4j.model.chat.listener.ChatModelResponseContext context) {
                        terminal.incrementAndGet();
                    }

                    @Override
                    public void onError(
                            dev.langchain4j.model.chat.listener.ChatModelErrorContext context) {
                        terminal.incrementAndGet();
                    }
                };
        StreamingChatModel failing = createFailingModel(counting);
        assertThrows(
                IllegalStateException.class,
                () ->
                        failing.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("this message will fail"))
                                        .build(),
                                new StreamingChatResponseHandler() {
                                    @Override
                                    public void onCompleteResponse(
                                            dev.langchain4j.model.chat.response.ChatResponse
                                                    response) {}

                                    @Override
                                    public void onError(Throwable error) {}
                                }));
        assertEquals(1, requests.get(), "onRequest fires before the synchronous rejection");
        assertEquals(0, terminal.get(), "no terminal listener event on a synchronous rejection");
    }
}
