package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.MultiModal;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Gemma 4 multimodal end-to-end: text GGUF + mmproj sidecar, image and audio content through the
 * native codec (media lowers to wrapped embeddings batches). Model-gated: assume-skips when either
 * GGUF is absent.
 */
@Tag("integration")
class Gemma4MediaIT {

    static final Path MODEL = ModelFixture.GEMMA4_E2B_QAT_Q4.path();
    static final Path MMPROJ = ModelFixture.GEMMA4_E2B_MMPROJ.path();

    // Audio: the 12B mmproj carries the gemma4ua audio adapter (the E2B sidecar is vision-only).
    static final Path AUDIO_MODEL = ModelFixture.GEMMA4_12B_QAT_Q4.path();
    static final Path AUDIO_MMPROJ = ModelFixture.GEMMA4_12B_QAT_MMPROJ.path();

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        Assumptions.assumeTrue(Files.exists(MMPROJ), "mmproj not found: " + MMPROJ);
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .mediaProjector(MMPROJ)
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .build();
    }

    @Test
    void mediaTokenEstimateMatchesTheBill() throws Exception {
        // the delta law: adding an image to a message must raise the ESTIMATE by exactly what it
        // raises the BILLED prompt tokens - scaffold cancels out of the difference, so this pins
        // the plan-based media count against ground truth end to end
        var estimator = model.tokenCountEstimator();
        UserMessage textOnly = UserMessage.from(TextContent.from("Describe this in one word."));
        UserMessage withImage =
                UserMessage.from(
                        ImageContent.from(solidPngB64(Color.BLUE), "image/png"),
                        TextContent.from("Describe this in one word."));
        int estimateDelta =
                estimator.estimateTokenCountInMessage(withImage)
                        - estimator.estimateTokenCountInMessage(textOnly);
        int billedWith =
                model.chat(ChatRequest.builder().messages(withImage).maxOutputTokens(1).build())
                        .tokenUsage()
                        .inputTokenCount();
        int billedWithout =
                model.chat(ChatRequest.builder().messages(textOnly).maxOutputTokens(1).build())
                        .tokenUsage()
                        .inputTokenCount();
        // the image also adds its two framing marker tokens to the prompt (scaffold, deliberately
        // outside the estimate) - allow exactly that
        int billedDelta = billedWith - billedWithout;
        assertTrue(
                billedDelta - estimateDelta >= 0 && billedDelta - estimateDelta <= 4,
                "estimate delta " + estimateDelta + " vs billed delta " + billedDelta);
        // the small test PNG plans to a small grid (e.g. 5x5 = 25) - the LAW above is the
        // assertion; this floor only guards against the media count silently becoming zero
        assertTrue(estimateDelta > 0, "image positions must be counted: " + estimateDelta);
    }

    @Test
    void describesImage() throws Exception {
        ChatResponse r =
                model.chat(
                        UserMessage.from(
                                ImageContent.from(solidPngB64(Color.RED), "image/png"),
                                TextContent.from(
                                        "What single color fills this image?"
                                                + " Answer with one word.")));
        assertNotNull(r.aiMessage().text());
        assertTrue(
                r.aiMessage().text().toLowerCase().contains("red"),
                "expected 'red' in: " + r.aiMessage().text());
    }

    @Test
    void consumesAudio() {
        Assumptions.assumeTrue(Files.exists(AUDIO_MODEL), "model not found: " + AUDIO_MODEL);
        Assumptions.assumeTrue(Files.exists(AUDIO_MMPROJ), "mmproj not found: " + AUDIO_MMPROJ);
        JinferChatModel audioModel =
                JinferChatModel.builder()
                        .modelPath(AUDIO_MODEL)
                        .mediaProjector(AUDIO_MMPROJ)
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .build();
        Assumptions.assumeTrue(
                engineModel(audioModel) instanceof MultiModal mm
                        && mm.modalities().contains(Media.Audio.class),
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
                                                        "Describe this audio in one sentence.")))
                                .build());
        assertNotNull(r.aiMessage().text());
        assertTrue(!r.aiMessage().text().isBlank());
    }

    @Test
    void imageInCachedPrompt() throws Exception {
        // the image lives in the CACHED prompt: decoded + prefilled once at view creation, its
        // prefill positions restored per request (the media ENCODER still runs to fingerprint)
        JinferChatModel scene =
                model.withCachedPrompt(
                        List.of(
                                UserMessage.from(
                                        ImageContent.from(solidPngB64(Color.BLUE), "image/png"),
                                        TextContent.from("This image is the reference scene."))),
                        List.of());
        ChatResponse first =
                scene.chat(
                        UserMessage.from("What single color fills the reference scene? One word."));
        assertTrue(
                first.aiMessage().text().toLowerCase().contains("blue"), first.aiMessage().text());
        // second request through the view: the image's prefill positions restore from blocks
        ChatResponse second = scene.chat(UserMessage.from("Is the scene dark or bright?"));
        assertTrue(!second.aiMessage().text().isBlank());
        String stats = model.engine.promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
    }

    @Test
    void toolRoundTrip() {
        var weather =
                dev.langchain4j.agent.tool.ToolSpecification.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .parameters(
                                dev.langchain4j.model.chat.request.json.JsonObjectSchema.builder()
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
                                        dev.langchain4j.data.message.ToolExecutionResultMessage
                                                .from(call.id(), call.name(), "18C, sunny"))
                                .toolSpecifications(weather)
                                .build());
        assertNotNull(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());

        // toolChoice REQUIRED: gemma's <|tool_call> marker seeds the reply and the prefix-pin
        // grammar guarantees the called NAME is an offered tool - a statement, not even a
        // question, with a decoy tool in the mix
        var decoy =
                dev.langchain4j.agent.tool.ToolSpecification.builder()
                        .name("get_time")
                        .description("Get the current time for a timezone")
                        .parameters(
                                dev.langchain4j.model.chat.request.json.JsonObjectSchema.builder()
                                        .addStringProperty("timezone")
                                        .build())
                        .build();
        ChatResponse forced =
                model.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("I live in Munich."))
                                .toolSpecifications(weather, decoy)
                                .toolChoice(dev.langchain4j.model.chat.request.ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                forced.aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + forced.aiMessage());
        String forcedName = forced.aiMessage().toolExecutionRequests().get(0).name();
        assertTrue(
                forcedName.equals("get_weather") || forcedName.equals("get_time"),
                "pinned to an offered tool, got: " + forcedName);
    }

    /** A solid 224x224 PNG as base64 (the shared image fixture). */
    private static String solidPngB64(Color color) throws java.io.IOException {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 224, 224);
        g.dispose();
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        return Base64.getEncoder().encodeToString(png.toByteArray());
    }

    private static Object engineModel(JinferChatModel m) {
        // the loaded LanguageModel implements MultiModal for gemma4
        return m.engine.loaded.model();
    }

    /** A mono 16-bit PCM WAV of a sine tone, built in memory. */
    static byte[] toneWav(double hz, double seconds, int rate) {
        int n = (int) (seconds * rate);
        byte[] pcm = new byte[n * 2];
        for (int i = 0; i < n; i++) {
            short s = (short) (Math.sin(2 * Math.PI * hz * i / rate) * 12000);
            pcm[i * 2] = (byte) s;
            pcm[i * 2 + 1] = (byte) (s >> 8);
        }
        var out = new ByteArrayOutputStream();
        try {
            var data = new java.io.DataOutputStream(out);
            data.writeBytes("RIFF");
            data.writeInt(Integer.reverseBytes(36 + pcm.length));
            data.writeBytes("WAVEfmt ");
            data.writeInt(Integer.reverseBytes(16));
            data.writeShort(Short.reverseBytes((short) 1)); // PCM
            data.writeShort(Short.reverseBytes((short) 1)); // mono
            data.writeInt(Integer.reverseBytes(rate));
            data.writeInt(Integer.reverseBytes(rate * 2));
            data.writeShort(Short.reverseBytes((short) 2));
            data.writeShort(Short.reverseBytes((short) 16));
            data.writeBytes("data");
            data.writeInt(Integer.reverseBytes(pcm.length));
            data.write(pcm);
        } catch (java.io.IOException impossible) {
            throw new AssertionError(impossible);
        }
        return out.toByteArray();
    }
}
