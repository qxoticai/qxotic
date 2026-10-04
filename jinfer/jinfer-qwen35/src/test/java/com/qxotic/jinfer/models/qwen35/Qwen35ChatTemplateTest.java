package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.MediaEncodingCache;
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
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class Qwen35ChatTemplateTest {
    private static Tokenizer tokenizer;
    private static String jinja;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        Path path = TestModels.require("hf.co/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf");
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
                        List.of(
                                Message.user("2+2?"),
                                Message.assistant("<think>old</think>\nFour"),
                                Message.user("3+3?")));
        for (boolean thinking : new boolean[] {false, true}) {
            for (List<Message> messages : cases) {
                int[] expected = render(messages, List.of(), thinking);
                for (int capacity : new int[] {1, 7, 512}) {
                    Conversation conversation = new Conversation(messages, List.of(), thinking);
                    assertArrayEquals(
                            expected,
                            encode(new Qwen35ChatTemplate(tokenizer), conversation, capacity));
                }
            }
        }
    }

    @Test
    void toolsCallsAndResultsMatchTheCheckpointTemplate() {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "get_weather");
        function.put("description", "Get weather");
        function.put("parameters", Map.of("type", "object"));
        Tool weather = new Tool("get_weather", Map.of("type", "function", "function", function));
        Message call =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.ToolCall("", "get_weather", Map.of("city", "Paris"))));
        Message result = new Message(Role.TOOL, List.of(new Content.ToolResult("", "18C, sunny")));
        List<Message> messages = List.of(Message.user("Weather?"), call, result);
        Conversation conversation = new Conversation(messages, List.of(weather), false);

        List<Map<String, Object>> mapped = new ArrayList<>();
        mapped.add(Map.of("role", "user", "content", "Weather?"));
        mapped.add(
                Map.of(
                        "role",
                        "assistant",
                        "content",
                        "",
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
        mapped.add(Map.of("role", "tool", "content", "18C, sunny"));
        int[] expected = renderMapped(mapped, List.of(weather.definition()), false);

        assertArrayEquals(expected, encode(new Qwen35ChatTemplate(tokenizer), conversation, 64));
    }

    @Test
    void parserRecognizesTheNativeFunctionWire() {
        ReplyParser parser = new Qwen35ChatTemplate(tokenizer).parser(tokenizer);
        Message reply =
                ReplyParser.parse(
                        parser,
                        SpecialTokens.encode(
                                tokenizer,
                                "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
                                        + "</parameter>\n</function>\n</tool_call><|im_end|>"));
        Content.ToolCall call =
                assertInstanceOf(Content.ToolCall.class, reply.content().getFirst());
        assertEquals("get_weather", call.name());
        assertEquals("Paris", call.arguments().get("city"));
    }

    @Test
    void rejectsShapesTheOfficialTemplateRejects() {
        Qwen35ChatTemplate template = new Qwen35ChatTemplate(tokenizer);
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, new Conversation(List.of()), 32));
        assertThrows(
                UnsupportedConversation.class,
                () ->
                        encode(
                                template,
                                new Conversation(
                                        List.of(Message.user("hi"), Message.system("late"))),
                                32));
        Message mediaSystem =
                new Message(
                        Role.SYSTEM,
                        List.of(
                                new Content.Media(
                                        new Media.Image(new float[] {0, 0, 0}, 1, 1, 3))));
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, new Conversation(List.of(mediaSystem)), 32));
        Tool tool = new Tool("noop", Map.of("name", "noop"));
        assertThrows(
                UnsupportedConversation.class,
                () ->
                        encode(
                                template,
                                new Conversation(
                                        List.of(Message.system("system")), List.of(tool), false),
                                32));
    }

    @Test
    void historyImagesReplayFromTheMediaCache() {
        ContentKey key = new ContentKey("image:test");
        Message image =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Media(
                                        new Media.Image(new float[] {0, 0, 0}, 1, 1, 3), key)));
        AtomicInteger projections = new AtomicInteger();
        try (Arena arena = Arena.ofConfined()) {
            Qwen35ChatTemplate template =
                    new Qwen35ChatTemplate(tokenizer, new CountingVision(arena, projections));
            MediaEncodingCache cache = new MediaEncodingCache();
            List<Message> turns = new ArrayList<>(List.of(image));
            for (int turn = 0; turn < 3; turn++) {
                template.encode(new Conversation(turns), 4, cache, ignored -> {});
                turns.add(Message.assistant("ok"));
                turns.add(Message.user("and now?"));
            }
        }
        assertEquals(1, projections.get());
    }

    private record CountingVision(Arena arena, AtomicInteger projections) implements Multimodal {
        @Override
        @SuppressWarnings("unchecked")
        public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
            if (modality != Media.Image.class) return Optional.empty();
            return Optional.of(
                    new MediaProjector<>() {
                        @Override
                        public int positions(R source) {
                            return 2;
                        }

                        @Override
                        public void project(
                                R source, int maxChunkSize, Consumer<MemoryView<?>> sink) {
                            projections.incrementAndGet();
                            sink.accept(Views.allocateF32(MemoryAllocators.ofArena(arena), 2, 3));
                        }
                    });
        }
    }

    private static int[] render(
            List<Message> messages, List<Map<String, Object>> tools, boolean thinking) {
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

    private static int[] encode(
            ChatTemplate template, Conversation conversation, int batchCapacity) {
        List<Batch> batches = new ArrayList<>();
        template.encode(conversation, batchCapacity, batches::add);
        return Batch.tokenIds(batches);
    }
}
