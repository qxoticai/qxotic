package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.service.AiServices;
import java.awt.Color;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The meaty showcase: a fully local multi-model agent. LFM2.5 (fast, tool-capable) is the brain
 * running langchain4j's automatic tool loop; Gemma 4 (vision + audio) is its eyes and ears, exposed
 * AS TOOLS. Two GGUFs, one JVM, zero cloud: the agent looks at images and listens to recordings by
 * calling the second model, keeps notes, and answers grounded across turns.
 *
 * <p>Model-gated (assume-skips without the GGUFs). The README's "local agent" section is this test.
 */
@Tag("integration")
class LocalAgentIT {

    private static final String BRAIN_REF =
            "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";
    private static final String EYES_REF =
            "hf.co/unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    private static final String EYES_MMPROJ_REF =
            "hf.co/unsloth/gemma-4-12B-it-qat-GGUF/mmproj-F32.gguf";

    /** Gemma 4 behind two tools: the brain never sees pixels or samples, only descriptions. */
    static class Senses {
        final ChatModel gemma;
        final List<String> log = new ArrayList<>();

        Senses(ChatModel gemma) {
            this.gemma = gemma;
        }

        @Tool("Look at an image file and answer a question about it")
        public String lookAt(
                @P("absolute path of the image file") String path,
                @P("what to look for or answer") String question) {
            log.add("lookAt(" + path + ")");
            System.err.println("  [tool] lookAt(" + path + ", \"" + question + "\")");
            if (log.size() > 6) return "Tool budget exhausted - answer from what you already know.";
            return gemma.chat(
                            ChatRequest.builder()
                                    .messages(
                                            UserMessage.from(
                                                    ImageContent.from(base64(path), "image/png"),
                                                    TextContent.from(question)))
                                    .build())
                    .aiMessage()
                    .text();
        }

        @Tool("Listen to an audio file and answer a question about it")
        public String listenTo(
                @P("absolute path of the audio file") String path,
                @P("what to listen for") String question) {
            log.add("listenTo(" + path + ")");
            System.err.println("  [tool] listenTo(" + path + ", \"" + question + "\")");
            if (log.size() > 6) return "Tool budget exhausted - answer from what you already know.";
            return gemma.chat(
                            ChatRequest.builder()
                                    .messages(
                                            UserMessage.from(
                                                    AudioContent.from(base64(path), "audio/wav"),
                                                    TextContent.from(question)))
                                    .build())
                    .aiMessage()
                    .text();
        }

        private static String base64(String path) {
            try {
                return Base64.getEncoder().encodeToString(Files.readAllBytes(Path.of(path)));
            } catch (Exception e) {
                return "unreadable: " + e.getMessage();
            }
        }
    }

    interface Agent {
        String chat(String message);
    }

    static Agent agent;
    static Senses senses;
    static JinferChatModel brain;
    static JinferChatModel eyes;

    @BeforeAll
    static void wire() {
        brain =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(BRAIN_REF))
                        .contextLength(8192)
                        .maxOutputTokens(512)
                        .build();
        eyes =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(EYES_REF))
                        .companion("media", TestModels.require(EYES_MMPROJ_REF))
                        .contextLength(4096)
                        .maxOutputTokens(256)
                        .build();
        senses = new Senses(eyes);
        agent =
                AiServices.builder(Agent.class)
                        .chatModel(brain)
                        .tools(senses)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                        .build();
    }

    @AfterAll
    static void unload() {
        if (brain != null) brain.close();
        if (eyes != null) eyes.close();
    }

    @Test
    void seesHearsRemembers() throws Exception {
        // scene: a "traffic light" picture and a tone recording. The dir name is FIXED: the path
        // appears verbatim in the prompts, and a random path would make every run a different
        // trajectory despite greedy sampling + fixed seed (observed flaky).
        Path dir = Path.of(System.getProperty("java.io.tmpdir"), "local-agent-demo");
        Files.createDirectories(dir);
        Path photo = dir.resolve("sign.png");
        ImageIO.write(trafficLight(), "png", photo.toFile());
        Path memo = dir.resolve("memo.wav");
        Files.write(memo, Gemma4MediaIT.toneWav(440, 1.0, 16000));

        String first =
                agent.chat(
                        "Look at "
                                + photo
                                + " and tell me the color of the TOP lamp of the traffic light.");
        System.out.println("AGENT> " + first);
        assertTrue(senses.log.stream().anyMatch(c -> c.startsWith("lookAt")), "used eyes");
        assertTrue(first.toLowerCase().contains("red"), first);

        String second =
                agent.chat("Now listen to " + memo + " - is it speech, music, or something else?");
        System.out.println("AGENT> " + second);
        assertTrue(senses.log.stream().anyMatch(c -> c.startsWith("listenTo")), "used ears");

        String third = agent.chat("Summarize everything you observed for me, one line each.");
        System.out.println("AGENT> " + third);
        assertTrue(
                third.toLowerCase().contains("red") || third.toLowerCase().contains("light"),
                third);
    }

    /** A 224x224 traffic light: dark housing, red/yellow/green lamps top to bottom. */
    private static BufferedImage trafficLight() {
        var img = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 224, 224);
        g.setColor(Color.DARK_GRAY);
        g.fillRoundRect(82, 20, 60, 184, 18, 18);
        g.setColor(Color.RED);
        g.fillOval(94, 34, 36, 36);
        g.setColor(Color.YELLOW);
        g.fillOval(94, 94, 36, 36);
        g.setColor(Color.GREEN);
        g.fillOval(94, 154, 36, 36);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
        g.dispose();
        return img;
    }
}
