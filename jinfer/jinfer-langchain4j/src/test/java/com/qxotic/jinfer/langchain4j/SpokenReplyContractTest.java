package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Modality;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The output-modalities contract on a model without a voice, model-free: allowed is not required,
 * audio alone is refused naming the reason, speech beside tools is refused by the engine's rule,
 * and a text reply streams no raw audio events.
 */
class SpokenReplyContractTest {

    private JinferChatModel model;

    @BeforeEach
    void load() {
        model = ChatFixtures.builder().build();
    }

    @AfterEach
    void close() {
        if (model != null) model.close();
    }

    private static ChatRequest request(JinferChatRequestParameters.Builder parameters) {
        return ChatRequest.builder()
                .messages(UserMessage.from("Hello"))
                .parameters(parameters.build())
                .build();
    }

    @Test
    void textAndAudioAllowedIsAnsweredInTextByAModelWithoutAVoice() {
        AiMessage reply =
                model.chat(
                                request(
                                        JinferChatRequestParameters.builder()
                                                .outputModalities(Modality.TEXT, Modality.AUDIO)))
                        .aiMessage();
        assertNotNull(reply.text());
        assertTrue(reply.audios().isEmpty(), "text it has, speech it has not: " + reply);
    }

    @Test
    void audioAloneIsRefusedNamingTheReason() {
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.chat(
                                        request(
                                                JinferChatRequestParameters.builder()
                                                        .outputModalities(Modality.AUDIO))));
        assertTrue(e.getMessage().contains("cannot speak"), e.getMessage());
    }

    @Test
    void speechBesideToolsIsRefusedBeforeTheModelRuns() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        request(
                                                JinferChatRequestParameters.builder()
                                                        .outputModalities(
                                                                Modality.TEXT, Modality.AUDIO)
                                                        .toolSpecifications(
                                                                ToolSpecification.builder()
                                                                        .name("weather")
                                                                        .description("weather")
                                                                        .build()))));
        assertTrue(e.getMessage().contains("cannot offer tools"), e.getMessage());
    }

    @Test
    void aTextReplyStreamsNoRawAudioEvents() throws Exception {
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        List<Object> raw = new CopyOnWriteArrayList<>();
        StringBuilder text = new StringBuilder();
        model.streaming()
                .chat(
                        request(
                                JinferChatRequestParameters.builder()
                                        .outputModalities(Modality.TEXT, Modality.AUDIO)),
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                text.append(partial);
                            }

                            @Override
                            public void onUnmappedRawEvent(Object event) {
                                raw.add(event);
                            }

                            @Override
                            public void onCompleteResponse(ChatResponse response) {
                                done.complete(response);
                            }

                            @Override
                            public void onError(Throwable error) {
                                done.completeExceptionally(error);
                            }
                        });
        AiMessage reply = done.get(30, TimeUnit.SECONDS).aiMessage();
        assertEquals(reply.text(), text.toString(), "streamed = the reply's text");
        assertTrue(raw.isEmpty(), "no speech, no raw event: " + raw);
        assertTrue(reply.audios().isEmpty());
    }
}
