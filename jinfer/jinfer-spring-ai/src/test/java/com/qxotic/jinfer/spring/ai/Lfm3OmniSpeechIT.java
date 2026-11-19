package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Modality;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * LFM3-Omni's spoken replies through the chat model, as a Spring application meets them: the
 * allowed output modalities on the options, the speech as a WAV {@link MediaPart} beside the
 * transcript, whole from {@code call} and a part per stretch from {@code stream}, and text alone
 * unless audio is allowed. The model's own speech recognition is the judge of what it said.
 */
@Tag("integration")
class Lfm3OmniSpeechIT {

    static final String OMNI = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF:Q8_0";
    static final String OMNI_MMPROJ =
            "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf";

    private static final UserMessage FRANCE = new UserMessage("What is the capital of France?");
    private static final JinferChatOptions SPOKEN =
            JinferChatOptions.builder()
                    .outputModalities(Modality.TEXT, Modality.AUDIO)
                    .maxTokens(400)
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
                        .options(JinferChatOptions.builder().maxTokens(64).seed(42L).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    /** Unset, the reply is text alone: only the request opens the voice. */
    @Test
    void textAloneUnlessAudioIsAllowed() {
        AssistantMessage out = model.call(new Prompt(FRANCE)).getResult().getOutput();
        assertTrue(out.getText().contains("Paris"), out.getText());
        assertTrue(speech(out).isEmpty(), "no media part: " + out.getParts());
    }

    /** Allowed, the reply speaks: one WAV media part after the transcript, saying what it wrote. */
    @Test
    void aSpokenReplyIsAWavMediaPartSayingWhatItWrote() throws IOException {
        ChatResponse r = model.call(new Prompt(List.of(FRANCE), SPOKEN));
        AssistantMessage out = r.getResult().getOutput();
        String text = out.getText();
        List<MediaPart> speech = speech(out);
        assertEquals(1, speech.size(), "one clip: " + out.getParts());
        assertTrue(out.getParts().indexOf(speech.get(0)) > 0, "after the transcript");
        assertEquals(JinferMappings.WAV, speech.get(0).media().getMimeType());
        Media.Audio clip = AudioCodec.decode(speech.get(0).media().getDataAsByteArray());
        String heard = hear(clip);
        System.out.printf("wrote: %s%nheard: %s (%.2f s)%n", text, heard, seconds(clip));
        assertTrue(seconds(clip) > 0.5, "speech");
        assertTrue(text.contains("Paris"), text);
        assertTrue(heard.contains("paris"), heard);
        assertTrue(out.getReasoning().isEmpty(), "a spoken reply does not reason");
    }

    /**
     * Streamed, the transcript is the text deltas and the speech a media chunk per stretch, all
     * under one response id; the final chunk carries neither. Joined, the chunks are the clip
     * {@code call} returns for the same seeded request.
     */
    @Test
    void aStreamedReplyIsTranscriptDeltasAndMediaChunks() throws IOException {
        List<ChatResponse> chunks =
                model.stream(new Prompt(List.of(FRANCE), SPOKEN))
                        .collectList()
                        .block(Duration.ofMinutes(5));
        assertNotNull(chunks);
        String id = chunks.get(0).getMetadata().getId();
        StringBuilder text = new StringBuilder();
        List<Media.Audio> stretches = new ArrayList<>();
        for (ChatResponse c : chunks) {
            assertEquals(id, c.getMetadata().getId(), "one id per response");
            AssistantMessage out = c.getResult().getOutput();
            text.append(out.getText());
            for (MediaPart part : speech(out))
                stretches.add(AudioCodec.decode(part.media().getDataAsByteArray()));
        }
        ChatResponse last = chunks.get(chunks.size() - 1);
        assertTrue(speech(last.getResult().getOutput()).isEmpty(), "the final chunk has no media");
        assertFalse(
                model.replies.recall(ChatFixtures.aggregate(chunks)).isEmpty(),
                "streamed = the reply's text");
        assertTrue(!stretches.isEmpty(), "no speech streamed");
        Media.Audio clip = Media.Audio.concat(stretches);
        String heard = hear(clip);
        System.out.printf(
                "streamed %d stretches, %.2f s; wrote: %s%nheard: %s%n",
                stretches.size(), seconds(clip), text, heard);
        assertTrue(heard.contains("paris"), heard);

        // ChatClient's view is the aggregator's: the transcript, the audio left to a subscriber
        String content =
                ChatClient.create(model).prompt(new Prompt(List.of(FRANCE), SPOKEN)).stream()
                        .content()
                        .collect(StringBuilder::new, StringBuilder::append)
                        .block(Duration.ofMinutes(5))
                        .toString();
        assertTrue(content.contains("Paris"), content);
    }

    /** The engine's refusals reach the caller in its words. */
    @Test
    void speechRefusesWhatItCannotHonor() {
        IllegalArgumentException tools =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                List.of(FRANCE),
                                                SPOKEN.mutate()
                                                        .toolCallbacks(List.of(noopTool()))
                                                        .build())));
        assertTrue(tools.getMessage().contains("cannot offer tools"), tools.getMessage());
        UnsupportedOperationException image =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                List.of(FRANCE),
                                                SPOKEN.mutate()
                                                        .outputModalities(Modality.IMAGE)
                                                        .build())));
        assertTrue(image.getMessage().contains("cannot produce"), image.getMessage());
    }

    /** Without its voice attached, the model answers in text and audio alone is refused. */
    @Test
    void withoutTheProjectorAudioIsRefusedNamingTheCompanion() {
        try (var bare =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(OMNI))
                        .contextCapacity(2048)
                        .options(JinferChatOptions.builder().maxTokens(32).build())
                        .build()) {
            AssistantMessage chosen = bare.call(new Prompt(FRANCE, SPOKEN)).getResult().getOutput();
            assertTrue(speech(chosen).isEmpty(), "text and audio allowed: text it has");
            assertTrue(chosen.getText().contains("Paris"), chosen.getText());
            UnsupportedOperationException e =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    bare.call(
                                            new Prompt(
                                                    List.of(FRANCE),
                                                    SPOKEN.mutate()
                                                            .outputModalities(Modality.AUDIO)
                                                            .build())));
            assertTrue(e.getMessage().contains("cannot speak"), e.getMessage());
            assertTrue(e.getMessage().contains("'media' companion"), e.getMessage());
        }
    }

    private static List<MediaPart> speech(AssistantMessage out) {
        List<MediaPart> parts = new ArrayList<>();
        for (MessagePart part : out.getParts()) if (part instanceof MediaPart m) parts.add(m);
        return parts;
    }

    /** What Omni's own ears make of {@code clip}, through the same chat model. */
    private static String hear(Media.Audio clip) {
        ChatResponse r =
                model.call(
                        new Prompt(
                                List.of(
                                        new SystemMessage("Perform ASR in English."),
                                        UserMessage.builder()
                                                .part(MediaPart.of(JinferMappings.audio(clip)))
                                                .build()),
                                JinferChatOptions.builder()
                                        .maxTokens(64)
                                        .temperature(0.0)
                                        .build()));
        return r.getResult().getOutput().getText().toLowerCase();
    }

    private static double seconds(Media.Audio audio) {
        return audio.pcm().length / (double) audio.sampleRate();
    }

    private static ToolCallback noopTool() {
        ToolDefinition def =
                DefaultToolDefinition.builder()
                        .name("weather")
                        .description("weather")
                        .inputSchema("{\"type\":\"object\"}")
                        .build();
        return new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
                return def;
            }

            @Override
            public String call(String toolInput) {
                return "";
            }
        };
    }
}
