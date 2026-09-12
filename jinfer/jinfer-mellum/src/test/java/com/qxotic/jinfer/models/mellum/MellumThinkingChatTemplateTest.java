package com.qxotic.jinfer.models.mellum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The Thinking checkpoint's template: the reasoning walk-back and the thinking-off prefix. */
@Tag("integration")
final class MellumThinkingChatTemplateTest {
    static final String MODEL_REF =
            "hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M/Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf";

    private static Tokenizer tokenizer;
    private static String jinja;
    private static MellumChatTemplate template;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        Path path = TestModels.require(MODEL_REF);
        try (FileChannel file = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            jinja = gguf.getString("tokenizer.chat_template");
        }
        assertTrue(jinja.contains("enable_thinking"));
        template = new MellumChatTemplate(tokenizer, true);
    }

    @Test
    void thinkingOnLeavesTheSpanToTheModelAndOffPreClosesIt() {
        assertEquals(ChatTemplate.ThinkingPolicy.OPTIONAL, template.thinkingPolicy());
        List<Message> messages =
                List.of(Message.system("Be brief."), Message.user("Why is the sky blue?"));
        int[] on = encode(new Conversation(messages, List.of(), true));
        int[] off = encode(new Conversation(messages, List.of(), false));
        assertArrayEquals(render(messages, List.of(), true), on);
        assertArrayEquals(render(messages, List.of(), false), off);
        int[] prefix =
                template.encode(new Conversation(messages, List.of(), false), 64, b -> {})
                        .replyPrefix()
                        .toArray();
        assertEquals("<think>\n\n</think>\n\n", tokenizer.decode(IntSequence.of(prefix)));
        assertArrayEquals(prefix, java.util.Arrays.copyOfRange(off, on.length, off.length));
    }

    @Test
    void reasoningIsEchoedOnlyAfterTheLastQuery() {
        Message thought =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning(
                                        List.of(new Content.Text("\nTwo plus two.\n")),
                                        IntSequence.empty()),
                                new Content.Text("\nFour")));
        List<List<Message>> cases =
                List.of(
                        // an earlier turn drops its reasoning, the turn after the last query keeps
                        // it
                        List.of(Message.user("2+2?"), thought, Message.user("Sure?"), thought),
                        // reasoning spelled inline in the text, the template's own split
                        List.of(
                                Message.user("2+2?"),
                                Message.assistant("<think>\nhmm\n</think>\n\nFour"),
                                Message.user("Sure?"),
                                Message.assistant("<think>\nhmm\n</think>\n\nFour")),
                        // a bare last assistant turn after the query still gets an empty span
                        List.of(Message.user("2+2?"), Message.assistant("Four")),
                        // a no-reasoning turn after the query that is not last is rendered bare
                        List.of(
                                Message.user("2+2?"),
                                Message.assistant("Four"),
                                Message.assistant("Four again")));
        for (List<Message> messages : cases)
            for (boolean thinking : new boolean[] {true, false})
                assertArrayEquals(
                        render(messages, List.of(), thinking),
                        encode(new Conversation(messages, List.of(), thinking)),
                        messages.toString());
    }

    @Test
    void toolRoundsKeepTheReasoningOfTheCallingTurn() {
        Tool weather =
                new Tool(
                        "get_weather",
                        Map.of("type", "function", "function", Map.of("name", "get_weather")));
        Message call =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning(
                                        List.of(new Content.Text("Need the weather.")),
                                        IntSequence.empty()),
                                new Content.ToolCall("", "get_weather", Map.of("city", "Paris"))));
        Message result = new Message(Role.TOOL, List.of(new Content.ToolResult("", "18C")));
        List<Message> messages = List.of(Message.user("Weather in Paris?"), call, result);
        List<Map<String, Object>> mapped = new ArrayList<>();
        mapped.add(Map.of("role", "user", "content", "Weather in Paris?"));
        mapped.add(
                Map.of(
                        "role",
                        "assistant",
                        "content",
                        "<think>Need the weather.</think>",
                        "tool_calls",
                        List.of(
                                Map.of(
                                        "type",
                                        "function",
                                        "function",
                                        Map.of(
                                                "name",
                                                "get_weather",
                                                "arguments",
                                                Map.of("city", "Paris"))))));
        mapped.add(Map.of("role", "tool", "content", "18C"));
        for (boolean thinking : new boolean[] {true, false})
            assertArrayEquals(
                    renderMapped(mapped, List.of(weather.definition()), thinking),
                    encode(new Conversation(messages, List.of(weather), thinking)));
    }

    /** The whole-render's view of a message: reasoning wrapped in the think markers. */
    private static int[] render(
            List<Message> messages, List<Map<String, Object>> tools, boolean thinking) {
        List<Map<String, Object>> mapped = new ArrayList<>();
        for (Message m : messages) {
            StringBuilder content = new StringBuilder();
            for (Content part : m.content()) {
                if (part instanceof Content.Text t) content.append(t.text());
                else if (part instanceof Content.Reasoning r)
                    content.append("<think>").append(r.text()).append("</think>");
            }
            mapped.add(Map.of("role", m.role().name(), "content", content.toString()));
        }
        return renderMapped(mapped, tools, thinking);
    }

    private static int[] renderMapped(
            List<Map<String, Object>> messages, List<Map<String, Object>> tools, boolean thinking) {
        String rendered =
                JinjaRenderer.template(jinja)
                        .render(
                                Map.of(
                                        "messages",
                                        messages,
                                        "tools",
                                        tools,
                                        "add_generation_prompt",
                                        true,
                                        "enable_thinking",
                                        thinking));
        return SpecialTokens.encode(tokenizer, rendered).toArray();
    }

    private static int[] encode(Conversation conversation) {
        List<Batch> batches = new ArrayList<>();
        template.encode(conversation, 7, batches::add);
        return Batch.tokenIds(batches);
    }
}
