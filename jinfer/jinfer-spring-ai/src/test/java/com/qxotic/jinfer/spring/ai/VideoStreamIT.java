package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.util.MimeType;

/**
 * A streamed video request on Gemma 4 E2B samples its clip once, not once more for the eager
 * validation. The sampler is the ffmpeg seam: here it counts, and hands back two solid frames, so
 * the case needs no codec and no fixture file.
 */
@Tag("integration")
class VideoStreamIT {

    @Test
    void aStreamedVideoIsSampledOnce() throws Exception {
        Media.Video clip =
                new Media.Video(
                        List.of(
                                new Media.Video.Frame(solid(Color.RED), Duration.ZERO),
                                new Media.Video.Frame(solid(Color.BLUE), Duration.ofSeconds(1))));
        AtomicInteger sampled = new AtomicInteger();
        Path placeholder = Files.createTempFile("clip", ".mp4");
        try (JinferChatModel model =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"))
                        .companionPath(
                                "media",
                                TestModels.require(
                                        "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"))
                        .videoSampler(
                                ignored -> {
                                    sampled.incrementAndGet();
                                    return clip;
                                })
                        .contextCapacity(4096)
                        .options(
                                JinferChatOptions.builder()
                                        .maxTokens(64)
                                        .temperature(0.0)
                                        .thinking(false)
                                        .build())
                        .build()) {
            Prompt prompt =
                    new Prompt(
                            UserMessage.builder()
                                    .text(
                                            "Name the solid colour of the first frame and of the"
                                                    + " last frame, in order, two words.")
                                    .media(
                                            new org.springframework.ai.content.Media(
                                                    MimeType.valueOf("video/mp4"),
                                                    placeholder.toUri()))
                                    .build());
            String answer =
                    model.stream(prompt)
                            .map(r -> r.getResult().getOutput().getText())
                            .collect(Collectors.joining())
                            .block(Duration.ofMinutes(5));
            assertEquals(1, sampled.get(), "the clip was sampled again");
            assertFalse(answer.isBlank());
        } finally {
            Files.deleteIfExists(placeholder);
        }
    }

    private static Media.Image solid(Color color) throws IOException {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 224, 224);
        g.dispose();
        var png = new ByteArrayOutputStream();
        ImageIO.write(img, "png", png);
        return ImageCodec.decode(png.toByteArray());
    }
}
