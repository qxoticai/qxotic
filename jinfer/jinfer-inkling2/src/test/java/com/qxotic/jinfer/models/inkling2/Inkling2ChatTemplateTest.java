package com.qxotic.jinfer.models.inkling2;

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
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The native codec against the checkpoint's own Jinja template, token for token. */
@Tag("integration")
final class Inkling2ChatTemplateTest {
    static final String MODEL_REF = "hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf";

    private static Tokenizer tokenizer;
    private static String jinja;
    private static Inkling2ChatTemplate template;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        Path path = TestModels.require(MODEL_REF);
        try (FileChannel file = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            jinja = gguf.getString("tokenizer.chat_template");
        }
        template =
                new Inkling2ChatTemplate(
                        tokenizer,
                        new Multimodal() { // no mmproj
                            @Override
                            public <R extends Media> Optional<MediaProjector<R>> projector(
                                    Class<R> modality) {
                                return Optional.empty();
                            }
                        });
    }

    @Test
    void plainPromptsMatchTheCheckpointTemplate() {
        List<List<Map<String, Object>>> cases =
                List.of(
                        List.of(turn("user", "Hello")),
                        List.of(turn("system", "Be concise."), turn("user", "Unicode ñ漢字")),
                        List.of(turn("system", "Only a system turn.")),
                        List.of(turn("user", "  padded  \n")),
                        List.of(turn("user", "")),
                        List.of(
                                turn("user", "2+2?"),
                                turn("assistant", "Four"),
                                turn("user", "3+3?"),
                                turn("system", "late system"),
                                turn("assistant", ""),
                                turn("user", "4+4?")));
        for (List<Map<String, Object>> turns : cases) {
            List<Message> messages = new ArrayList<>();
            for (Map<String, Object> turn : turns)
                messages.add(
                        new Message(
                                new Role((String) turn.get("role")), (String) turn.get("content")));
            for (boolean thinking : new boolean[] {true, false}) {
                int[] expected = render(turns, List.of(), thinking);
                for (int capacity : new int[] {1, 7, 512})
                    assertArrayEquals(
                            expected,
                            encode(new Conversation(messages, List.of(), thinking), capacity));
            }
        }
    }

    @Test
    void reasoningToolsCallsAndResultsMatchTheCheckpointTemplate() {
        Tool weather = weather();
        Tool time =
                new Tool(
                        "get_time",
                        Map.of("type", "function", "function", Map.of("name", "get_time")));
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("zone", Map.of("name", "Europe/Paris"));
        nested.put("city", "Paris \"Île\"");
        nested.put("days", 2L);
        Message thought =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning("They want both."),
                                new Content.Text("Checking."),
                                new Content.ToolCall("call_1", "get_weather", nested),
                                new Content.ToolCall("call_2", "get_time", Map.of())));
        Message bareCall =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.ToolCall("", "get_weather", Map.of("city", "Oslo"))));
        Message silent =
                new Message(Role.ASSISTANT, List.of(new Content.Reasoning("Nothing to add.")));
        List<Message> messages =
                List.of(
                        Message.system("Answer briefly."),
                        Message.user("Weather and time?"),
                        thought,
                        // answered out of order: the header names the call the id points at
                        new Message(Role.TOOL, List.of(new Content.ToolResult("call_2", "12:00"))),
                        new Message(
                                Role.TOOL, List.of(new Content.ToolResult("call_1", "18C, sunny"))),
                        Message.user("And Oslo?"),
                        bareCall,
                        new Message(Role.TOOL, List.of(new Content.ToolResult("", "3C"))),
                        silent,
                        Message.user("Thanks"));
        // as llama.cpp cbb1c1c50 renders it: the template sorts a call's argument keys with
        // dictsort, which the Jinja oracle of the other test does not implement
        String expected =
                "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get weather\","
                        + "\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":"
                        + "{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":"
                        + "[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":"
                        + "\"\",\"name\":\"get_time\",\"parameters\":{},\"type\":\"function\"}]"
                        + "<|end_message|><|message_system|><|content_text|>Answer briefly."
                        + "<|end_message|><|message_system|><|content_text|>Thinking effort"
                        + " level: 0.9<|end_message|><|message_user|><|content_text|>Weather and"
                        + " time?<|end_message|><|message_model|><|content_thinking|>They want"
                        + " both.<|end_message|><|message_model|><|content_text|>Checking."
                        + "<|end_message|><|message_model|>get_weather"
                        + "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
                        + "{\"city\":\"Paris \\\"Île\\\"\",\"days\":2,\"zone\":{\"name\":"
                        + "\"Europe/Paris\"}}}<|end_message|><|message_model|>get_time"
                        + "<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{}}"
                        + "<|end_message|><|content_model_end_sampling|><|message_tool|>get_time"
                        + "<|content_text|>12:00<|end_message|><|message_tool|>get_weather"
                        + "<|content_text|>18C, sunny<|end_message|><|message_user|>"
                        + "<|content_text|>And Oslo?<|end_message|><|message_model|>get_weather"
                        + "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
                        + "{\"city\":\"Oslo\"}}<|end_message|><|content_model_end_sampling|>"
                        + "<|message_tool|>get_weather<|content_text|>3C<|end_message|>"
                        + "<|message_model|><|content_thinking|>Nothing to add.<|end_message|>"
                        + "<|content_model_end_sampling|><|message_user|><|content_text|>Thanks"
                        + "<|end_message|>";
        for (int capacity : new int[] {3, 64, 4096})
            assertArrayEquals(
                    SpecialTokens.encode(tokenizer, expected).toArray(),
                    encode(new Conversation(messages, List.of(weather, time), true), capacity));
    }

    @Test
    void parserReadsTheBlocksOfAReply() {
        assertEquals(ChatTemplate.ThinkingPolicy.OPTIONAL, template.thinkingPolicy());
        Message reply =
                ReplyParser.parse(
                        template.parser(tokenizer),
                        SpecialTokens.encode(
                                tokenizer,
                                "<|message_model|><|content_thinking|>They want Paris."
                                        + "<|end_message|><|message_model|><|content_text|>Let me"
                                        + " check.<|end_message|><|message_model|>get_weather"
                                        + "<|content_invoke_tool_json|>{\"name\":\"get_weather\","
                                        + "\"args\":{\"city\":\"Paris\",\"days\":2}}"
                                        + "<|end_message|><|content_model_end_sampling|>"));
        assertEquals("They want Paris.", reply.reasoning());
        assertEquals("Let me check.", reply.text());
        Content.ToolCall call = assertInstanceOf(Content.ToolCall.class, reply.content().getLast());
        assertEquals("get_weather", call.name());
        assertEquals("Paris", call.arguments().get("city"));
        assertEquals(2L, call.arguments().get("days"));
        assertTrue(template.forcedCall(List.of(weather())).isPresent());
        assertTrue(template.constrainedReply("root ::= \"yes\" | \"no\"").isPresent());
    }

    @Test
    void rejectsMedia() {
        Message image =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Media(
                                        new Media.Image(new float[] {0, 0, 0}, 1, 1, 3))));
        assertThrows(
                UnsupportedConversation.class, () -> encode(new Conversation(List.of(image)), 32));
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
                        Map.of("city", Map.of("type", "string"), "days", Map.of("type", "integer")),
                        "required",
                        List.of("city")));
        return new Tool("get_weather", Map.of("type", "function", "function", function));
    }

    private static Map<String, Object> turn(String role, String content) {
        Map<String, Object> turn = new LinkedHashMap<>();
        turn.put("role", role);
        turn.put("content", content);
        return turn;
    }

    private static int[] render(
            List<Map<String, Object>> messages, List<Map<String, Object>> tools, boolean thinking) {
        Map<String, Object> bindings = new LinkedHashMap<>();
        bindings.put("messages", messages);
        bindings.put("tools", tools);
        // by name: "high" is the template's default, 0.9
        bindings.put("reasoning_effort", thinking ? "high" : "none");
        return SpecialTokens.encode(tokenizer, JinjaRenderer.template(jinja).render(bindings))
                .toArray();
    }

    private static int[] encode(Conversation conversation, int batchCapacity) {
        List<Batch> batches = new ArrayList<>();
        template.encode(conversation, batchCapacity, batches::add);
        return Batch.tokenIds(batches);
    }
}
