package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.MultiModal;
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

    static final Path MODEL =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
    static final Path MMPROJ =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");

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
    void describesImage() throws Exception {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 224, 224);
        g.dispose();
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);

        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                ImageContent.from(
                                                        Base64.getEncoder()
                                                                .encodeToString(png.toByteArray()),
                                                        "image/png"),
                                                TextContent.from(
                                                        "What single color fills this image?"
                                                                + " Answer with one word.")))
                                .build());
        assertNotNull(r.aiMessage().text());
        assertTrue(
                r.aiMessage().text().toLowerCase().contains("red"),
                "expected 'red' in: " + r.aiMessage().text());
    }

    @Test
    void consumesAudio() {
        Assumptions.assumeTrue(
                engineModel() instanceof MultiModal mm
                        && mm.modalities().contains(Media.Audio.class),
                "mmproj carries no audio adapter");
        byte[] wav = toneWav(440, 1.0, 16000);
        ChatResponse r =
                model.chat(
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

    private static Object engineModel() {
        // the loaded LanguageModel implements MultiModal for gemma4
        return modelUnderTest().loaded.model();
    }

    private static JinferEngine modelUnderTest() {
        try {
            var f = JinferChatModel.class.getDeclaredField("engine");
            f.setAccessible(true);
            return (JinferEngine) f.get(model);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    /** A mono 16-bit PCM WAV of a sine tone, built in memory. */
    private static byte[] toneWav(double hz, double seconds, int rate) {
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
