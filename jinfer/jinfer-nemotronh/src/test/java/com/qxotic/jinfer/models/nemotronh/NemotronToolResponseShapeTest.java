package com.qxotic.jinfer.models.nemotronh;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * The tool-RESPONSE turn renders its text for BOTH wire shapes: the typed {@link Part.ToolResult}
 * (framework adapters) and the plain {@link Part.Text} the server lowers {@code {role:"tool",
 * content}} to. The Text shape regressed silently once - the template rendered an EMPTY {@code
 * <tool_response>} and the model re-called its tool forever (caught live on the OpenAI server, not
 * by the oracle, whose fixtures used the typed shape).
 */
final class NemotronToolResponseShapeTest {

    private static final String RESULT = "{\"temp_c\": 7, \"condition\": \"light rain\"}";

    @Test
    void bothToolResultShapesRenderTheResultText() throws Exception {
        Path model = ModelFixture.NEMOTRON_30B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model));
        GGUF gguf;
        try (var ch = FileChannel.open(model, StandardOpenOption.READ)) {
            gguf = ModelLoader.readGguf(ch, model.toString());
        }
        Tokenizer tokenizer = Tokenizers.fromGGUF(gguf);
        NemotronHTurnTemplate template = new NemotronHTurnTemplate(tokenizer);
        Tool tool =
                new Tool(
                        "get_weather",
                        "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}");

        for (Part resultPart :
                List.of(new Part.ToolResult("call_0", RESULT), new Part.Text(RESULT))) {
            List<Batch> prompt =
                    template.encode(
                            new Conversation(
                                    List.of(
                                            Message.user("Weather in Zurich?"),
                                            new Message(
                                                    Role.ASSISTANT,
                                                    List.of(
                                                            new Part.ToolCall(
                                                                    "call_0",
                                                                    "get_weather",
                                                                    Map.of("city", "Zurich")))),
                                            new Message(Role.TOOL, List.of(resultPart))),
                                    List.of(tool),
                                    false,
                                    ""));
            StringBuilder text = new StringBuilder();
            for (Batch b : prompt) {
                if (b.input() instanceof Batch.Input.Tokens t) {
                    text.append(tokenizer.decode(t.ids()));
                }
            }
            assertTrue(
                    text.toString().contains(RESULT),
                    resultPart.getClass().getSimpleName()
                            + " tool result must render its text; got:\n"
                            + text);
        }
    }
}
