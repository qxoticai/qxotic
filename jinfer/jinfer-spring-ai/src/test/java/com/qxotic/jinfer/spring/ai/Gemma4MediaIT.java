package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.util.MimeType;

/**
 * Multimodal end-to-end against Gemma 4 12B with its mmproj sidecar (vision + audio encoders):
 * images and audio enter as embeddings, never as text. Model-gated via {@link TestModels}. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class Gemma4MediaIT {

    static final Path MODEL =
            TestModels.require(
                    "hf.co/unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf");
    static final Path MMPROJ =
            TestModels.require("hf.co/unsloth/gemma-4-12B-it-qat-GGUF/mmproj-F32.gguf");

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .companion("media", MMPROJ)
                        .contextLength(4096)
                        .options(JinferChatOptions.builder().maxTokens(512).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void describesImage() throws Exception {
        ChatResponse r =
                model.call(
                        new Prompt(
                                UserMessage.builder()
                                        .text(
                                                "What single color fills this image? Answer with"
                                                        + " one word.")
                                        .media(
                                                org.springframework.ai.content.Media.builder()
                                                        .mimeType(MimeType.valueOf("image/png"))
                                                        .data(solidPng(Color.RED))
                                                        .build())
                                        .build()));
        String text = r.getResult().getOutput().getText();
        assertNotNull(text);
        assertTrue(text.toLowerCase(Locale.ROOT).contains("red"), "expected 'red' in: " + text);
    }

    @Test
    void consumesAudio() {
        Assumptions.assumeTrue(
                model.engine.loaded().model() instanceof Multimodal mm
                        && mm.projector(Media.Audio.class).isPresent(),
                "mmproj carries no audio adapter");
        ChatResponse r =
                model.call(
                        new Prompt(
                                UserMessage.builder()
                                        .text("Describe this audio in one sentence.")
                                        .media(
                                                org.springframework.ai.content.Media.builder()
                                                        .mimeType(MimeType.valueOf("audio/wav"))
                                                        .data(toneWav(440, 1.0, 16000))
                                                        .build())
                                        .build()));
        String text = r.getResult().getOutput().getText();
        assertNotNull(text);
        assertTrue(!text.isBlank());
    }

    @Test
    void imageInCachedPrompt() throws Exception {
        // the image lives in the CACHED prompt: decoded + prefilled once at view creation, its
        // prefill positions restored per request (the media ENCODER still runs to fingerprint)
        JinferChatModel scene =
                model.withCachedPrompt(
                        List.of(
                                UserMessage.builder()
                                        .text("This image is the reference scene.")
                                        .media(
                                                org.springframework.ai.content.Media.builder()
                                                        .mimeType(MimeType.valueOf("image/png"))
                                                        .data(solidPng(Color.BLUE))
                                                        .build())
                                        .build()),
                        List.of());
        ChatResponse first =
                scene.call(
                        new Prompt(
                                new UserMessage(
                                        "What single color fills the reference scene? One word.")));
        assertTrue(
                first.getResult().getOutput().getText().toLowerCase(Locale.ROOT).contains("blue"),
                first.getResult().getOutput().getText());
        // second request through the view: the image's prefill positions restore from blocks
        ChatResponse second =
                scene.call(new Prompt(new UserMessage("Is the scene dark or bright?")));
        assertTrue(!second.getResult().getOutput().getText().isBlank());
        String stats = model.engine.promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
    }

    /** A solid 224x224 PNG (the shared image fixture). */
    private static byte[] solidPng(Color color) throws IOException {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 224, 224);
        g.dispose();
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        return png.toByteArray();
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
