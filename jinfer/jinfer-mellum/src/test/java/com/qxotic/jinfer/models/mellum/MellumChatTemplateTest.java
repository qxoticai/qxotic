package com.qxotic.jinfer.models.mellum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The native codec against the checkpoint's own Jinja template, token for token. */
@Tag("integration")
final class MellumChatTemplateTest {
    static final String MODEL_REF =
            "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0/Mellum2-12B-A2.5B-Instruct-Q8_0.gguf";

    private static Tokenizer tokenizer;
    private static String jinja;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        Path path = TestModels.require(MODEL_REF);
        try (FileChannel file = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            jinja = gguf.getString("tokenizer.chat_template");
        }
    }

    @Test
    void plainPromptsMatchTheCheckpointTemplate() {
        List<List<Message>> cases =
                List.of(
                        List.of(Message.user("Hello")),
                        List.of(Message.system("Be concise."), Message.user("Unicode ñ漢字")),
                        List.of(Message.user("  padded  \n")),
                        List.of(
                                Message.user("2+2?"),
                                Message.assistant("<think>old</think>\n\nFour"),
                                Message.user("3+3?"),
                                Message.system("late system"),
                                Message.user("4+4?")));
        for (List<Message> messages : cases) {
            int[] expected = render(messages, List.of());
            for (int capacity : new int[] {1, 7, 512}) {
                assertArrayEquals(
                        expected,
                        encode(
                                new MellumChatTemplate(tokenizer),
                                new Conversation(messages, List.of(), false),
                                capacity));
            }
        }
    }

    @Test
    void toolsCallsAndResultsMatchTheCheckpointTemplate() {
        Tool weather = weather();
        Tool time =
                new Tool(
                        "get_time",
                        Map.of("type", "function", "function", Map.of("name", "get_time")));
        Message twoCalls =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Text("Checking."),
                                new Content.ToolCall("", "get_weather", Map.of("city", "Paris")),
                                new Content.ToolCall("", "get_time", Map.of())));
        Message bareCall =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.ToolCall("", "get_weather", Map.of("city", "Oslo"))));
        Message result1 = new Message(Role.TOOL, List.of(new Content.ToolResult("", "18C, sunny")));
        Message result2 = new Message(Role.TOOL, List.of(new Content.ToolResult("", "12:00")));
        List<Message> messages =
                List.of(
                        Message.system("Answer briefly."),
                        Message.user("Weather and time?"),
                        twoCalls,
                        result1,
                        result2,
                        Message.user("And Oslo?"),
                        bareCall,
                        result1);
        Conversation conversation = new Conversation(messages, List.of(weather, time), false);

        List<Map<String, Object>> mapped = new ArrayList<>();
        mapped.add(Map.of("role", "system", "content", "Answer briefly."));
        mapped.add(Map.of("role", "user", "content", "Weather and time?"));
        mapped.add(
                Map.of(
                        "role",
                        "assistant",
                        "content",
                        "Checking.",
                        "tool_calls",
                        List.of(
                                call("get_weather", Map.of("city", "Paris")),
                                call("get_time", Map.of()))));
        mapped.add(Map.of("role", "tool", "content", "18C, sunny"));
        mapped.add(Map.of("role", "tool", "content", "12:00"));
        mapped.add(Map.of("role", "user", "content", "And Oslo?"));
        mapped.add(
                Map.of(
                        "role",
                        "assistant",
                        "content",
                        "",
                        "tool_calls",
                        List.of(call("get_weather", Map.of("city", "Oslo")))));
        mapped.add(Map.of("role", "tool", "content", "18C, sunny"));
        int[] expected = renderMapped(mapped, List.of(weather.definition(), time.definition()));

        for (int capacity : new int[] {3, 64, 4096})
            assertArrayEquals(
                    expected, encode(new MellumChatTemplate(tokenizer), conversation, capacity));
    }

    @Test
    void toolsWithoutASystemMessageMatchTheCheckpointTemplate() {
        Tool weather = weather();
        List<Message> messages = List.of(Message.user("Weather?"));
        int[] expected =
                renderMapped(
                        List.of(Map.of("role", "user", "content", "Weather?")),
                        List.of(weather.definition()));
        assertArrayEquals(
                expected,
                encode(
                        new MellumChatTemplate(tokenizer),
                        new Conversation(messages, List.of(weather), false),
                        64));
    }

    @Test
    void parserRecognizesTheJsonEnvelopeWire() {
        MellumChatTemplate template = new MellumChatTemplate(tokenizer);
        assertEquals(ChatTemplate.ThinkingPolicy.NONE, template.thinkingPolicy());
        Message reply =
                ReplyParser.parse(
                        template.parser(tokenizer),
                        SpecialTokens.encode(
                                tokenizer,
                                "Let me check.\n<tool_call>\n{\"name\": \"get_weather\","
                                        + " \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n"
                                        + "</tool_call><|im_end|>"));
        assertEquals("Let me check.\n", reply.text());
        Content.ToolCall call = assertInstanceOf(Content.ToolCall.class, reply.content().getLast());
        assertEquals("get_weather", call.name());
        assertEquals("Paris", call.arguments().get("city"));
        assertEquals(2L, call.arguments().get("days"));
        assertTrue(template.forcedCall(List.of(weather())).isPresent());
    }

    @Test
    void rejectsShapesTheTemplateCannotFrame() {
        MellumChatTemplate template = new MellumChatTemplate(tokenizer);
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, new Conversation(List.of()), 32));
        Message image =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Media(
                                        new Media.Image(new float[] {0, 0, 0}, 1, 1, 3))));
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, new Conversation(List.of(image)), 32));
    }

    private static Tool weather() {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "get_weather");
        function.put("description", "Get weather");
        function.put(
                "parameters",
                Map.of(
                        "type",
                        "object",
                        "properties",
                        Map.of("city", Map.of("type", "string")),
                        "required",
                        List.of("city")));
        return new Tool("get_weather", Map.of("type", "function", "function", function));
    }

    private static Map<String, Object> call(String name, Map<String, Object> arguments) {
        return Map.of("type", "function", "function", Map.of("name", name, "arguments", arguments));
    }

    private static int[] render(List<Message> messages, List<Map<String, Object>> tools) {
        List<Map<String, Object>> mapped =
                messages.stream()
                        .map(
                                message ->
                                        Map.<String, Object>of(
                                                "role",
                                                message.role().name(),
                                                "content",
                                                message.text()))
                        .toList();
        return renderMapped(mapped, tools);
    }

    private static int[] renderMapped(
            List<Map<String, Object>> messages, List<Map<String, Object>> tools) {
        String rendered =
                JinjaRenderer.template(jinja)
                        .render(
                                Map.of(
                                        "messages",
                                        messages,
                                        "tools",
                                        tools,
                                        "add_generation_prompt",
                                        true));
        return SpecialTokens.encode(tokenizer, rendered).toArray();
    }

    private static int[] encode(
            ChatTemplate template, Conversation conversation, int batchCapacity) {
        List<Batch> batches = new ArrayList<>();
        template.encode(conversation, batchCapacity, batches::add);
        return Batch.tokenIds(batches);
    }
}
