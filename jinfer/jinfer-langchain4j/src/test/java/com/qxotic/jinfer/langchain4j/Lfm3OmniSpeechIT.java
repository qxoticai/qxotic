package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Modality;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM3-Omni's spoken replies through the chat model, as a langchain4j application meets them: the
 * allowed output modalities on the request parameters, the speech as the {@code AiMessage}'s one
 * WAV audio beside the transcript, whole from {@code chat} and a stretch per raw event from the
 * streaming model, and text alone unless audio is allowed. The model's own speech recognition is
 * the judge of what it said.
 */
@Tag("integration")
class Lfm3OmniSpeechIT {

    static final String OMNI = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF:Q8_0";
    static final String OMNI_MMPROJ =
            "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf";

    private static final SystemMessage VOICE_CHAT =
            SystemMessage.from("Respond with interleaved text and audio.");
    private static final UserMessage FRANCE = UserMessage.from("What is the capital of France?");
    private static final JinferChatRequestParameters SPOKEN =
            JinferChatRequestParameters.builder()
                    .outputModalities(Modality.TEXT, Modality.AUDIO)
                    .maxOutputTokens(400)
                    .temperature(0.0)
                    .seed(42L)
                    .build();

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(OMNI))
                        .companionPath("media", TestModels.require(OMNI_MMPROJ))
                        .contextCapacity(4096)
                        .maxOutputTokens(64)
                        .seed(42L)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    /**
     * Unset, the reply is text alone: the voice's opener is masked, whatever the prompt asks. Asked
     * for audio it cannot give, Omni may say nothing at all (HF's reference does the same), so the
     * masked case checks the absence of speech only.
     */
    @Test
    void textAloneUnlessAudioIsAllowed() {
        AiMessage out = model.chat(FRANCE).aiMessage();
        assertTrue(out.text().contains("Paris"), out.text());
        assertTrue(out.audios().isEmpty(), "no audio: " + out);
        AiMessage asked = model.chat(VOICE_CHAT, FRANCE).aiMessage();
        assertTrue(asked.audios().isEmpty(), "no audio: " + asked);
    }

    /** Allowed, the reply speaks: one WAV audio beside the transcript, saying what it wrote. */
    @Test
    void aSpokenReplyIsAWavAudioSayingWhatItWrote() throws IOException {
        AiMessage out = model.chat(spoken(VOICE_CHAT, FRANCE)).aiMessage();
        String text = out.text();
        assertEquals(1, out.audios().size(), "one clip: " + out);
        Audio speech = out.audios().get(0);
        assertEquals("audio/wav", speech.mimeType());
        Media.Audio clip = AudioCodec.decode(speech.binaryData());
        String heard = hear(AudioContent.from(speech));
        System.out.printf("wrote: %s%nheard: %s (%.2f s)%n", text, heard, seconds(clip));
        assertTrue(seconds(clip) > 0.5, "speech");
        assertTrue(text.contains("Paris"), text);
        assertTrue(heard.contains("paris"), heard);
        assertNull(out.thinking(), "a spoken reply does not reason");
    }

    /**
     * Streamed, the transcript is the partial responses and the speech a raw event per stretch, the
     * engine's own clip; the complete response carries the whole clip, as {@code chat} does.
     * Joined, the stretches are that clip.
     */
    @Test
    void aStreamedReplyIsTranscriptDeltasAndRawAudioEvents() throws Exception {
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        StringBuilder text = new StringBuilder();
        List<Media.Audio> stretches = new CopyOnWriteArrayList<>();
        model.streaming()
                .chat(
                        spoken(VOICE_CHAT, FRANCE),
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                text.append(partial);
                            }

                            @Override
                            public void onUnmappedRawEvent(Object event) {
                                stretches.add((Media.Audio) event);
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
        AiMessage whole = done.get(5, TimeUnit.MINUTES).aiMessage();
        assertEquals(whole.text(), text.toString(), "streamed = the reply's text");
        assertTrue(!stretches.isEmpty(), "no speech streamed");
        Media.Audio clip = Media.Audio.concat(stretches);
        assertEquals(1, whole.audios().size(), "the complete response carries the clip");
        assertEquals(
                AudioCodec.decode(whole.audios().get(0).binaryData()).pcm().length,
                AudioCodec.decode(AudioCodec.wav(clip)).pcm().length,
                "the stretches joined are the clip");
        String heard =
                hear(
                        AudioContent.from(
                                Audio.builder()
                                        .binaryData(AudioCodec.wav(clip))
                                        .mimeType("audio/wav")
                                        .build()));
        System.out.printf(
                "streamed %d stretches, %.2f s; wrote: %s%nheard: %s%n",
                stretches.size(), seconds(clip), text, heard);
        assertTrue(heard.contains("paris"), heard);
    }

    /** The engine's refusals reach the caller in langchain4j's words. */
    @Test
    void speechRefusesWhatItCannotHonor() {
        IllegalArgumentException tools =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(VOICE_CHAT, FRANCE)
                                                .parameters(
                                                        JinferChatRequestParameters.builder()
                                                                .overrideWith(SPOKEN)
                                                                .toolSpecifications(
                                                                        ToolSpecification.builder()
                                                                                .name("weather")
                                                                                .description(
                                                                                        "weather")
                                                                                .build())
                                                                .build())
                                                .build()));
        assertTrue(tools.getMessage().contains("cannot offer tools"), tools.getMessage());
        UnsupportedFeatureException image =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(FRANCE)
                                                .parameters(
                                                        JinferChatRequestParameters.builder()
                                                                .overrideWith(SPOKEN)
                                                                .outputModalities(Modality.IMAGE)
                                                                .build())
                                                .build()));
        assertTrue(image.getMessage().contains("cannot produce"), image.getMessage());
    }

    /** Without its voice attached, the model answers in text and audio alone is refused. */
    @Test
    void withoutTheProjectorAudioIsRefusedNamingTheCompanion() {
        try (var bare =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(OMNI))
                        .contextCapacity(2048)
                        .maxOutputTokens(32)
                        .build()) {
            AiMessage chosen = bare.chat(spoken(FRANCE)).aiMessage();
            assertTrue(chosen.audios().isEmpty(), "text and audio allowed: text it has");
            assertTrue(chosen.text().contains("Paris"), chosen.text());
            UnsupportedFeatureException e =
                    assertThrows(
                            UnsupportedFeatureException.class,
                            () ->
                                    bare.chat(
                                            ChatRequest.builder()
                                                    .messages(FRANCE)
                                                    .parameters(
                                                            JinferChatRequestParameters.builder()
                                                                    .overrideWith(SPOKEN)
                                                                    .outputModalities(
                                                                            Modality.AUDIO)
                                                                    .build())
                                                    .build()));
            assertTrue(e.getMessage().contains("cannot speak"), e.getMessage());
            assertTrue(e.getMessage().contains("'media' companion"), e.getMessage());
        }
    }

    private static ChatRequest spoken(ChatMessage... messages) {
        return ChatRequest.builder().messages(messages).parameters(SPOKEN).build();
    }

    /** What Omni's own ears make of {@code speech}, through the same chat model. */
    private static String hear(AudioContent speech) {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        SystemMessage.from("Perform ASR in English."),
                                        UserMessage.from(speech))
                                .maxOutputTokens(64)
                                .temperature(0.0)
                                .build());
        return r.aiMessage().text().toLowerCase();
    }

    private static double seconds(Media.Audio audio) {
        return audio.pcm().length / (double) audio.sampleRate();
    }
}
