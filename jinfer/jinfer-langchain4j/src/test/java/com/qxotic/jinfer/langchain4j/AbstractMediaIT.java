package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The model-independent media contract, end-to-end against a real GGUF. A model instance is in
 * exactly one of two states, and the battery pins both lanes:
 *
 * <p><b>Media-capable</b> (a multimodal checkpoint WITH its companion attached - {@link
 * #mediaCompanion()} non-null): images decode, embed, and answer (solid-color fixtures, the
 * semantically reliable lane for small checkpoints); the estimator's media count matches the billed
 * tokens; an image inside a cached prompt survives the block round-trip.
 *
 * <p><b>Bare</b> (a text-only family, OR a vision-capable checkpoint loaded WITHOUT its companion):
 * media content is refused loudly with the remedy in the message - never silently dropped from the
 * prompt. A vision-capable checkpoint bare and a text-only family are the SAME contract at this
 * seam; the subclasses exist because they reach it through different template code (Gemma4's native
 * codec punts from {@code requireSupported}, LFM2's from its own).
 *
 * <p>Parameterized by model: each concrete subclass names one GGUF (assume-skips when absent) and,
 * for the capable cell, its companion.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class AbstractMediaIT {

    /** The GGUF this subclass runs against. */
    abstract Path modelPath();

    /** The media companion (mmproj) to attach, or null for a bare load. */
    Path mediaCompanion() {
        return null;
    }

    /** Whether this instance accepts media - the companion is the capability. */
    boolean supportsMedia() {
        return mediaCompanion() != null;
    }

    JinferChatModel model;

    @BeforeAll
    void load() {
        Assumptions.assumeTrue(Files.exists(modelPath()), "model not found: " + modelPath());
        var builder =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .contextLength(4096)
                        .maxOutputTokens(512);
        if (mediaCompanion() != null) {
            Assumptions.assumeTrue(
                    Files.exists(mediaCompanion()), "companion not found: " + mediaCompanion());
            builder.companionPath("media", mediaCompanion());
        }
        model = builder.build();
    }

    @AfterAll
    void unload() {
        if (model != null) model.close();
    }

    // ---- accept lane: a capable instance sees the image ----

    @Test
    void describesImage() throws Exception {
        Assumptions.assumeTrue(supportsMedia(), "bare instance: the fail lane covers it");
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
    void mediaTokenEstimateMatchesTheBill() throws Exception {
        // the delta law: adding an image to a message must raise the ESTIMATE by exactly what it
        // raises the BILLED prompt tokens - scaffold cancels out of the difference, so this pins
        // the plan-based media count against ground truth end to end
        Assumptions.assumeTrue(supportsMedia(), "bare instance: the fail lane covers it");
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
    void imageInCachedPrompt() throws Exception {
        // the image lives in the CACHED prompt: decoded + prefilled once at view creation, its
        // prefill positions restored per request (the media ENCODER still runs to fingerprint)
        Assumptions.assumeTrue(supportsMedia(), "bare instance: the fail lane covers it");
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

    // ---- fail lane: a bare instance refuses loudly, with the remedy ----

    @Test
    void imageIsRefusedLoudlyWithTheRecipe() throws Exception {
        // the silent-drop ban: a bare instance must not answer from a prompt the image was
        // dropped out of. The refusal names the content kind AND the way out.
        Assumptions.assumeFalse(supportsMedia(), "capable instance: the accept lane covers it");
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.chat(
                                        UserMessage.from(
                                                ImageContent.from(
                                                        solidPngB64(Color.GREEN), "image/png"),
                                                TextContent.from("What color is this?"))));
        assertTrue(e.getMessage().contains("image"), e.getMessage());
        assertTrue(e.getMessage().contains("not supported"), e.getMessage());
        assertTrue(
                e.getMessage().contains("companion"),
                "the remedy rides the refusal: " + e.getMessage());
    }

    @Test
    void audioIsRefusedLoudly() {
        Assumptions.assumeFalse(supportsMedia(), "capable instance: the accept lane covers it");
        String wav = Base64.getEncoder().encodeToString(toneWav(440, 0.2, 16000));
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.chat(
                                        UserMessage.from(
                                                dev.langchain4j.data.message.AudioContent.from(
                                                        wav, "audio/wav"),
                                                TextContent.from("Describe this audio."))));
        assertTrue(e.getMessage().contains("not supported"), e.getMessage());
    }

    // ---- shared fixtures ----

    /** A solid 224x224 PNG as base64 (the shared image fixture). */
    static String solidPngB64(Color color) throws IOException {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 224, 224);
        g.dispose();
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        return Base64.getEncoder().encodeToString(png.toByteArray());
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
            var data = new DataOutputStream(out);
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
        } catch (IOException impossible) {
            throw new AssertionError(impossible);
        }
        return out.toByteArray();
    }
}
