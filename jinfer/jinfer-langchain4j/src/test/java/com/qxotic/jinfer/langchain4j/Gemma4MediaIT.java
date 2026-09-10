package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.awt.Color;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * {@link AbstractMediaIT} against Gemma 4 multimodal end-to-end: text GGUF + mmproj sidecar, image
 * through the native codec (media lowers to wrapped embeddings batches). This subclass keeps the
 * two Gemma-specific lanes the battery cannot generalize: the 12B {@code gemma4ua} audio path and
 * the tool-call interplay. Model-gated: assume-skips when a GGUF is absent.
 */
class Gemma4MediaIT extends AbstractMediaIT {

    private static final String MODEL_REF =
            "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
    private static final String MMPROJ_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    // Exercise 12B's gemma4ua path; E2B/E4B use the separate gemma4a Conformer.
    private static final String AUDIO_MODEL_REF =
            "hf.co/unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    private static final String AUDIO_MMPROJ_REF =
            "hf.co/unsloth/gemma-4-12B-it-qat-GGUF/mmproj-F32.gguf";

    @Override
    Path modelPath() {
        return TestModels.require(MODEL_REF);
    }

    @Override
    Path mediaCompanion() {
        return TestModels.require(MMPROJ_REF);
    }

    @Test
    void consumesVideoAsTimestampedFrames() throws Exception {
        // the sampler is the ffmpeg seam: here it hands back two solid frames, red then blue,
        // so the case needs no codec and no fixture file
        Media.Video clip =
                new Media.Video(
                        List.of(
                                new Media.Video.Frame(
                                        ImageCodec.decode(
                                                Base64.getDecoder().decode(solidPngB64(Color.RED))),
                                        Duration.ZERO),
                                new Media.Video.Frame(
                                        ImageCodec.decode(
                                                Base64.getDecoder()
                                                        .decode(solidPngB64(Color.BLUE))),
                                        Duration.ofSeconds(1))));
        Path placeholder = Files.createTempFile("clip", ".mp4");
        try (JinferChatModel videoModel =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .companionPath("media", mediaCompanion())
                        .videoSampler(ignored -> clip)
                        .contextLength(4096)
                        .maxOutputTokens(64)
                        .temperature(0.0)
                        .thinking(false)
                        .build()) {
            String answer =
                    videoModel
                            .chat(
                                    UserMessage.from(
                                            VideoContent.from(placeholder.toUri().toString()),
                                            TextContent.from(
                                                    "Name the solid colour of the first frame and"
                                                            + " of the last frame, in order, two"
                                                            + " words.")))
                            .aiMessage()
                            .text()
                            .toLowerCase();
            assertTrue(
                    answer.indexOf("red") >= 0 && answer.indexOf("blue") > answer.indexOf("red"),
                    answer);
        } finally {
            Files.deleteIfExists(placeholder);
        }
    }

    @Test
    void consumesAudio() {
        try (JinferChatModel audioModel =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(AUDIO_MODEL_REF))
                        .companionPath("media", TestModels.require(AUDIO_MMPROJ_REF))
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        // greedy and seeded: sampled, the 12B asks for the file ~1 draw in 6
                        .temperature(0.0)
                        .seed(7L)
                        .build()) {
            Assumptions.assumeTrue(
                    engineModel(audioModel) instanceof Multimodal mm
                            && mm.projector(Media.Audio.class).isPresent(),
                    "mmproj carries no audio adapter");
            // eight seconds: the 12B hears four or more, three or less reads as no audio
            byte[] wav = toneWav(440, 8.0, 16000);
            ChatResponse r =
                    audioModel.chat(
                            ChatRequest.builder()
                                    .messages(
                                            UserMessage.from(
                                                    AudioContent.from(
                                                            Base64.getEncoder().encodeToString(wav),
                                                            "audio/wav"),
                                                    TextContent.from(
                                                            "Describe this audio in one"
                                                                    + " sentence.")))
                                    .build());
            String heard = r.aiMessage().text();
            assertNotNull(heard);
            assertTrue(!heard.isBlank());
            // heard, not merely said: "please provide the audio file" is non-blank too
            assertFalse(
                    heard.matches("(?is).*\\b(provide|upload|attach|share)\\b.*"),
                    "asked for the file instead: " + heard);
            assertTrue(
                    heard.matches(
                            "(?is).*\\b(tone|music|musical|sound|sounds|beep|pitch|note|synth|hum"
                                    + "|beat|melody|electronic)\\b.*"),
                    "no sound described: " + heard);
        }
    }

    @Test
    void toolRoundTrip() {
        var weather =
                ToolSpecification.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        ChatResponse first =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the"
                                                        + " get_weather tool."))
                                .toolSpecifications(weather)
                                .build());
        Assumptions.assumeTrue(
                first.aiMessage().hasToolExecutionRequests(),
                "model chose not to call the tool: " + first.aiMessage().text());
        var call = first.aiMessage().toolExecutionRequests().get(0);
        assertTrue("get_weather".equals(call.name()), call.name());
        assertTrue(call.arguments().contains("Paris"), call.arguments());
        ChatResponse second =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the"
                                                        + " get_weather tool."),
                                        first.aiMessage(),
                                        ToolExecutionResultMessage.from(
                                                call.id(), call.name(), "18C, sunny"))
                                .toolSpecifications(weather)
                                .build());
        assertNotNull(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());

        // toolChoice REQUIRED: gemma's <|tool_call> marker seeds the reply and the prefix-pin
        // grammar guarantees the called NAME is an offered tool - a statement, not even a
        // question, with a decoy tool in the mix
        var decoy =
                ToolSpecification.builder()
                        .name("get_time")
                        .description("Get the current time for a timezone")
                        .parameters(
                                JsonObjectSchema.builder().addStringProperty("timezone").build())
                        .build();
        ChatResponse forced =
                model.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("I live in Munich."))
                                .toolSpecifications(weather, decoy)
                                .toolChoice(ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                forced.aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + forced.aiMessage());
        String forcedName = forced.aiMessage().toolExecutionRequests().get(0).name();
        assertTrue(
                forcedName.equals("get_weather") || forcedName.equals("get_time"),
                "pinned to an offered tool, got: " + forcedName);
    }

    private static Object engineModel(JinferChatModel m) {
        // the loaded LanguageModel implements Multimodal for gemma4
        return m.engine.loaded().model();
    }
}
