package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import java.util.Base64;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * {@link AbstractMediaIT} against Gemma 4 multimodal end-to-end: text GGUF + mmproj sidecar, image
 * through the native codec (media lowers to wrapped embeddings batches). This subclass keeps the
 * two Gemma-specific lanes the battery cannot generalize: audio (the 12B mmproj carries the
 * gemma4ua audio adapter; the E2B sidecar is vision-only) and the tool-call interplay. Model-gated:
 * assume-skips when a GGUF is absent.
 */
class Gemma4MediaIT extends AbstractMediaIT {

    private static final String MODEL_REF =
            "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
    private static final String MMPROJ_REF = "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    // Audio: the 12B mmproj carries the gemma4ua audio adapter (the E2B sidecar is vision-only).
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
    void consumesAudio() {
        try (JinferChatModel audioModel =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(AUDIO_MODEL_REF))
                        .companion("media", TestModels.require(AUDIO_MMPROJ_REF))
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .build()) {
            Assumptions.assumeTrue(
                    engineModel(audioModel) instanceof Multimodal mm
                            && mm.projector(Media.Audio.class).isPresent(),
                    "mmproj carries no audio adapter");
            byte[] wav = toneWav(440, 1.0, 16000);
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
            assertNotNull(r.aiMessage().text());
            assertTrue(!r.aiMessage().text().isBlank());
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
