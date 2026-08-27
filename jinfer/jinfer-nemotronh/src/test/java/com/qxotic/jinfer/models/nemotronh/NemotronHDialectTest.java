package com.qxotic.jinfer.models.nemotronh;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
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

/** The codec follows each Nemotron checkpoint's own template, byte for byte. */
final class NemotronHDialectTest {

    private static final String CASCADE =
            "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q8_0";
    private static final String LIGHTNING =
            "hf.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_0";

    private record Checkpoint(Tokenizer tokenizer, String template) {
        static Checkpoint load(String ref) throws Exception {
            Path path = TestModels.require(ref);
            try (FileChannel file = FileChannel.open(path)) {
                GGUF gguf = ModelLoader.readGguf(file, path.toString());
                return new Checkpoint(
                        GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf),
                        gguf.getStringOrDefault("tokenizer.chat_template", ""));
            }
        }

        int[] jinja(
                List<Map<String, Object>> messages,
                List<Map<String, Object>> tools,
                boolean thinking) {
            Map<String, Object> context = new LinkedHashMap<>();
            context.put("messages", messages);
            if (tools != null) context.put("tools", tools);
            context.put("add_generation_prompt", true);
            context.put("enable_thinking", thinking);
            String rendered = JinjaRenderer.template(template).render(context);
            return SpecialTokens.encode(tokenizer, rendered).toArray();
        }

        int[] codec(
                NemotronHChatTemplate codec,
                List<Message> messages,
                List<Tool> tools,
                boolean thinking) {
            List<Batch> batches = new ArrayList<>();
            codec.encode(new Conversation(messages, tools, thinking, ""), 512, batches::add);
            return Batch.tokenIds(batches);
        }

        NemotronHChatTemplate own() {
            return new NemotronHChatTemplate(tokenizer, NemotronHChatTemplate.Dialect.of(template));
        }

        void assertParity(
                List<Map<String, Object>> maps,
                List<Map<String, Object>> toolMaps,
                List<Message> messages,
                List<Tool> tools,
                boolean thinking) {
            int[] expected = jinja(maps, toolMaps, thinking);
            int[] actual = codec(own(), messages, tools, thinking);
            assertEquals(tokenizer.decode(expected), tokenizer.decode(actual));
            assertArrayEquals(expected, actual);
        }
    }

    private static Map<String, Object> map(String role, String content) {
        return Map.of("role", role, "content", content);
    }

    private static final Map<String, Object> WEATHER_TOOL =
            Map.of(
                    "type",
                    "function",
                    "function",
                    Map.of(
                            "name",
                            "get_weather",
                            "description",
                            "Current weather",
                            "parameters",
                            Map.of(
                                    "type",
                                    "object",
                                    "properties",
                                    Map.of("city", Map.of("type", "string")))));

    private static void checkAll(Checkpoint c) {
        // no system turn: Cascade injects its sentence, 3.5 renders an empty system turn
        c.assertParity(
                List.of(map("user", "Hi")), null, List.of(Message.user("Hi")), List.of(), true);
        // an assistant turn after the last user keeps its reasoning; markers are the ids
        String answer = "<think>\nplan\n</think>\n4";
        c.assertParity(
                List.of(map("user", "2+2?"), map("assistant", answer)),
                null,
                List.of(Message.user("2+2?"), Message.assistant(answer)),
                List.of(),
                false);
        // a tool turn in a request that offers no tools is still the folded user turn
        c.assertParity(
                List.of(
                        map("user", "Weather?"),
                        map("assistant", "<think></think>Calling."),
                        map("tool", "{\"t\": 7}")),
                null,
                List.of(
                        Message.user("Weather?"),
                        Message.assistant("<think></think>Calling."),
                        new Message(Role.TOOL, List.of(new Content.Text("{\"t\": 7}")))),
                List.of(),
                true);
        // typed reasoning kept after the last user, tools offered: the dialect's think shape
        Map<String, Object> typed = new LinkedHashMap<>();
        typed.put("role", "assistant");
        typed.put("reasoning_content", "plan");
        typed.put("content", "42");
        c.assertParity(
                List.of(map("user", "2+2?"), typed),
                List.of(WEATHER_TOOL),
                List.of(
                        Message.user("2+2?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Content.Reasoning(
                                                List.of(new Content.Text("plan")), null),
                                        new Content.Text("42")))),
                List.of(new Tool("get_weather", WEATHER_TOOL)),
                false);
    }

    @Test
    @Tag("integration")
    void cascadeTwoMatchesItsTemplate() throws Exception {
        checkAll(Checkpoint.load(CASCADE));
    }

    @Test
    @Tag("integration")
    void lightningMatchesItsTemplate() throws Exception {
        Checkpoint c = Checkpoint.load(LIGHTNING);
        checkAll(c);
        // the Cascade-only codec renders a system sentence the 3.5 template never produces
        int[] cascadeShaped =
                c.codec(
                        new NemotronHChatTemplate(c.tokenizer),
                        List.of(Message.user("Hi")),
                        List.of(),
                        true);
        assertFalse(Arrays.equals(c.jinja(List.of(map("user", "Hi")), null, true), cascadeShaped));
    }
}
