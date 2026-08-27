package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The codec follows each LFM2.5 checkpoint's own template dialect, byte for byte. */
final class Lfm2DialectTest {

    private record Checkpoint(Tokenizer tokenizer, GGUF gguf, String template) {
        static Checkpoint load(String ref) throws Exception {
            Path path = TestModels.require(ref);
            try (FileChannel file = FileChannel.open(path)) {
                GGUF gguf = ModelLoader.readGguf(file, path.toString());
                Tokenizer tokenizer =
                        GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
                return new Checkpoint(
                        tokenizer, gguf, gguf.getStringOrDefault("tokenizer.chat_template", ""));
            }
        }

        int[] jinja(List<Map<String, Object>> messages) {
            String rendered =
                    JinjaRenderer.template(template)
                            .render(
                                    Map.of(
                                            "messages",
                                            messages,
                                            "add_generation_prompt",
                                            true,
                                            "bos_token",
                                            "<|startoftext|>",
                                            "eos_token",
                                            "<|im_end|>"));
            return SpecialTokens.encode(tokenizer, rendered).toArray();
        }

        /** The codec before dialects: one shape for every checkpoint. */
        int[] oldCodec(List<Message> messages) {
            List<Batch> batches = new ArrayList<>();
            new Lfm2ChatTemplate(tokenizer, Lfm2ChatTemplate.promptOpensThinking(template))
                    .encode(new Conversation(messages), 512, batches::add);
            return Batch.tokenIds(batches);
        }

        int[] codec(List<Message> messages) {
            List<Batch> batches = new ArrayList<>();
            Lfm2ChatTemplate.fromGguf(tokenizer, gguf)
                    .encode(new Conversation(messages), 512, batches::add);
            return Batch.tokenIds(batches);
        }
    }

    @Test
    @Tag("integration")
    void theSmallCheckpointKeepsTheLastAssistantsThinking() throws Exception {
        // LFM2.5-350M keys on last_assistant_index: the last assistant turn keeps its thinking
        // even with a user turn after it (2.6B/8B strip everything before the last user)
        Checkpoint c = Checkpoint.load("hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0");
        String answer = "<think>two and two</think>Four.";
        int[] expected =
                c.jinja(
                        List.of(
                                Map.of("role", "user", "content", "2+2?"),
                                Map.of("role", "assistant", "content", answer),
                                Map.of("role", "user", "content", "And 3+3?")));
        List<Message> messages =
                List.of(Message.user("2+2?"), Message.assistant(answer), Message.user("And 3+3?"));
        assertEquals(c.tokenizer.decode(expected), c.tokenizer.decode(c.codec(messages)));
        assertArrayEquals(expected, c.codec(messages));
        assertFalse(
                Arrays.equals(expected, c.oldCodec(messages)),
                "the single-dialect codec strips what the 350M template keeps");
    }

    @Test
    @Tag("integration")
    void theEightBRendersArgumentsRawWithPythonLists() throws Exception {
        // LFM2.5-8B's format_arg_value: strings raw between single quotes, lists via | string
        // (Python repr); 2.6B escapes and uses JSON
        Checkpoint c = Checkpoint.load("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0");
        Map<String, Object> arguments = new LinkedHashMap<>();
        arguments.put("q", "it's here");
        arguments.put("tags", List.of("a", "b"));
        Map<String, Object> call =
                Map.of("function", Map.of("name", "search", "arguments", arguments));
        Map<String, Object> assistant = new LinkedHashMap<>();
        assistant.put("role", "assistant");
        assistant.put("content", "");
        assistant.put("tool_calls", List.of(call));
        int[] expected =
                c.jinja(
                        List.of(
                                Map.of("role", "user", "content", "Search."),
                                assistant,
                                Map.of("role", "tool", "content", "[]")));
        List<Message> messages =
                List.of(
                        Message.user("Search."),
                        new Message(
                                Role.ASSISTANT,
                                List.of(new Content.ToolCall("", "search", arguments))),
                        new Message(Role.TOOL, List.of(new Content.ToolResult("", "[]"))));
        assertEquals(c.tokenizer.decode(expected), c.tokenizer.decode(c.codec(messages)));
        assertArrayEquals(expected, c.codec(messages));
        assertFalse(
                Arrays.equals(expected, c.oldCodec(messages)),
                "the single-dialect codec escapes what the 8B template renders raw");
    }
}
