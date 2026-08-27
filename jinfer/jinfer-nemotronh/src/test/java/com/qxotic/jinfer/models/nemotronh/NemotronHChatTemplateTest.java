package com.qxotic.jinfer.models.nemotronh;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Both framework and server tool-result shapes must retain their payload in the prompt. */
class NemotronHChatTemplateTest {

    private static final String RESULT = "{\"temp_c\": 7, \"condition\": \"light rain\"}";

    @Test
    void typedAndPlainToolResultsBothRenderTheirText() throws Exception {
        Path path =
                TestModels.require("hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q8_0");
        Tokenizer tokenizer;
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        NemotronHChatTemplate template = new NemotronHChatTemplate(tokenizer);
        Tool tool =
                new Tool(
                        "get_weather",
                        Map.of("type", "function", "function", Map.of("name", "get_weather")));

        for (Content result :
                List.of(new Content.ToolResult("call_0", RESULT), new Content.Text(RESULT))) {
            Conversation conversation =
                    new Conversation(
                            List.of(
                                    Message.user("Weather in Zurich?"),
                                    new Message(
                                            Role.ASSISTANT,
                                            List.of(
                                                    new Content.ToolCall(
                                                            "call_0",
                                                            "get_weather",
                                                            Map.of("city", "Zurich")))),
                                    new Message(Role.TOOL, List.of(result))),
                            List.of(tool),
                            false,
                            "");
            ArrayList<Batch> batches = new ArrayList<>();
            template.encode(conversation, 512, batches::add);
            String prompt = tokenizer.decode(Batch.tokenIds(batches));
            assertTrue(
                    prompt.contains(RESULT),
                    result.getClass().getSimpleName() + " dropped the tool result:\n" + prompt);
        }
    }
}
