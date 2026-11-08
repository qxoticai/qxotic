package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyLanguage;
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
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Every conversation shape the native codec frames, against the checkpoint's own Jinja template
 * rendered and encoded the way llama.cpp does (special spellings parsed). Media rows are -1 on our
 * side and the template's placeholder is that many -1s on the reference side.
 */
@Tag("integration")
final class Lfm3ChatTemplateTest {

    static final String TEXT = "hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf";
    static final String OMNI = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q8_0.gguf";
    private static final int ROWS = 3;

    private record Checkpoint(Tokenizer tokenizer, String jinja) {}

    private static Checkpoint checkpoint(String ref) throws Exception {
        Path path = TestModels.require(ref);
        try (FileChannel file = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            return new Checkpoint(
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf),
                    gguf.getString("tokenizer.chat_template"));
        }
    }

    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    Map.of(
                            "type",
                            "function",
                            "function",
                            ordered(
                                    "name",
                                    "get_weather",
                                    "description",
                                    "Current weather for a city",
                                    "parameters",
                                    ordered(
                                            "type",
                                            "object",
                                            "properties",
                                            Map.of("city", Map.of("type", "string")),
                                            "required",
                                            List.of("city")))));

    @ParameterizedTest
    @ValueSource(strings = {TEXT, OMNI})
    void conversationsMatchTheCheckpointTemplate(String ref) throws Exception {
        Checkpoint c = checkpoint(ref);
        Map<String, Object> arguments =
                ordered(
                        "city",
                        "Paris",
                        "days",
                        3,
                        "metric",
                        true,
                        "where",
                        ordered("lat", 48.85, "tags", List.of("a", "b")),
                        "note",
                        null);
        Message call =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Text("Let me check."),
                                new Content.ToolCall("", "get_weather", arguments),
                                new Content.ToolCall("", "get_weather", Map.of("city", "Rome"))));
        Message bareCall =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.ToolCall("", "get_weather", Map.of("city", "Oslo"))));
        Message result = new Message(Role.TOOL, List.of(new Content.ToolResult("", "18C, sunny")));
        Message reasoned =
                new Message(
                        Role.ASSISTANT,
                        List.of(new Content.Reasoning("Add them."), new Content.Text("Four")));
        List<Conversation> cases =
                List.of(
                        conversation(Message.user("Hello")),
                        conversation(Message.system("Be concise."), Message.user("Unicode ñ漢字 ")),
                        conversation(Message.system(""), Message.user("empty system")),
                        conversation(
                                Message.user("2+2?"),
                                Message.assistant("<think>old</think>\n Four "),
                                Message.user("3+3?")),
                        conversation(Message.user("2+2?"), reasoned, Message.user("again")),
                        conversation(Message.user("2+2?"), reasoned),
                        conversation(
                                Message.user("2+2?"), Message.assistant("<think>kept</think>4")),
                        conversation(
                                Message.user("hi"),
                                Message.assistant("hello"),
                                Message.system("late")),
                        new Conversation(
                                List.of(Message.user("Weather?")), List.of(WEATHER), false),
                        new Conversation(
                                List.of(
                                        Message.system("Be brief."),
                                        Message.user("Weather?"),
                                        call,
                                        result),
                                List.of(WEATHER),
                                false),
                        new Conversation(
                                List.of(
                                        Message.user("Weather?"),
                                        bareCall,
                                        result,
                                        reasoned,
                                        Message.user("ok")),
                                List.of(WEATHER),
                                false));
        for (Conversation conversation : cases) {
            for (boolean thinking : new boolean[] {false, true}) {
                Conversation turn =
                        new Conversation(conversation.messages(), conversation.tools(), thinking);
                int[] expected = render(c, turn);
                for (int capacity : new int[] {1, 7, 512})
                    assertArrayEquals(
                            expected,
                            encode(new Lfm3ChatTemplate(c.tokenizer(), null), turn, capacity),
                            turn.toString());
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {TEXT, OMNI})
    void imagesAndAudioReplaceTheirPlaceholders(String ref) throws Exception {
        Checkpoint c = checkpoint(ref);
        Media.Image image = new Media.Image(new float[] {0, 0, 0}, 1, 1, 3);
        Media.Audio audio = new Media.Audio(new float[] {0f}, 16_000, 1);
        List<Message> messages = new ArrayList<>();
        messages.add(
                new Message(
                        Role.USER,
                        List.of(new Content.Text("What is this? "), new Content.Media(image))));
        if (ref.equals(OMNI))
            messages.add(
                    new Message(
                            Role.USER,
                            List.of(new Content.Media(audio), new Content.Text("Transcribe."))));
        try (Arena arena = Arena.ofConfined()) {
            Lfm3ChatTemplate template = new Lfm3ChatTemplate(c.tokenizer(), new FakeTowers(arena));
            Conversation conversation = new Conversation(messages, List.of(), false);
            assertArrayEquals(render(c, conversation), encode(template, conversation, 512));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {TEXT, OMNI})
    void theParserReadsTheModelsOwnWire(String ref) throws Exception {
        Checkpoint c = checkpoint(ref);
        ReplyParser parser = new Lfm3ChatTemplate(c.tokenizer(), null).parser(c.tokenizer());
        Message reply =
                ReplyParser.parse(
                        parser,
                        SpecialTokens.encode(
                                c.tokenizer(),
                                "<think>The user wants weather.</think>Checking.<tool_call>\n"
                                        + "<function=get_weather>\n<parameter=city>\nParis\n"
                                        + "</parameter>\n<parameter=days>\n3\n</parameter>\n"
                                        + "</function>\n</tool_call><|im_end|>"));
        assertEquals(
                "The user wants weather.",
                assertInstanceOf(Content.Reasoning.class, reply.content().get(0)).text());
        assertEquals(
                "Checking.", assertInstanceOf(Content.Text.class, reply.content().get(1)).text());
        Content.ToolCall call = assertInstanceOf(Content.ToolCall.class, reply.content().get(2));
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris", "days", 3L), Map.copyOf(call.arguments()));
    }

    @ParameterizedTest
    @ValueSource(strings = {TEXT})
    void aForcedCallAdmitsTheModelsOwnWire(String ref) throws Exception {
        Checkpoint c = checkpoint(ref);
        ReplyLanguage.Selection forced =
                new Lfm3ChatTemplate(c.tokenizer(), null)
                        .forcedCall(List.of(WEATHER))
                        .orElseThrow();
        ReplyLanguage.Walk walk = forced.walk();
        walk.seed(IntSequence.of(forced.forcedPrefix()));
        // the call as the checkpoint writes it: <function=, <parameter= ... are its added tokens
        IntSequence wire =
                SpecialTokens.encode(
                        c.tokenizer(),
                        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                                + "</function>\n</tool_call>");
        int size = c.tokenizer().vocabulary().size();
        for (int token : wire.toArray()) {
            MemoryView<MemorySegment> logits =
                    Views.fromFloatArray(MemoryAllocators.ofArena(Arena.ofAuto()), new float[size]);
            walk.maskLogits(logits);
            assertEquals(
                    0f,
                    Views.getFloat(logits, token, "logits"),
                    () -> c.tokenizer().decode(new int[] {token}));
            walk.feed(token);
        }
        Content.ToolCall call =
                assertInstanceOf(Content.ToolCall.class, walk.finish().content().getFirst());
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
    }

    @ParameterizedTest
    @ValueSource(strings = {TEXT})
    void mediaWithoutATowerAndMediaInTheSystemTurnAreRefused(String ref) throws Exception {
        Checkpoint c = checkpoint(ref);
        Lfm3ChatTemplate template = new Lfm3ChatTemplate(c.tokenizer(), null);
        Content.Media image = new Content.Media(new Media.Image(new float[] {0, 0, 0}, 1, 1, 3));
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, conversation(new Message(Role.USER, List.of(image))), 32));
        assertThrows(
                UnsupportedConversation.class,
                () ->
                        encode(
                                template,
                                conversation(
                                        new Message(Role.SYSTEM, List.of(image)),
                                        Message.user("hi")),
                                32));
        assertThrows(
                UnsupportedConversation.class,
                () -> encode(template, new Conversation(List.of()), 32));
    }

    // --- the reference side: the GGUF's Jinja, encoded as llama.cpp does -----------------------

    private static int[] render(Checkpoint c, Conversation conversation) {
        List<Map<String, Object>> messages = new ArrayList<>();
        for (Message m : conversation.messages()) messages.add(jinjaMessage(m));
        Map<String, Object> vars = new LinkedHashMap<>();
        vars.put("messages", messages);
        vars.put("tools", conversation.tools().stream().map(Tool::definition).toList());
        vars.put("add_generation_prompt", true);
        vars.put("enable_thinking", conversation.thinking());
        vars.put("bos_token", "<|startoftext|>");
        String rendered = JinjaRenderer.template(c.jinja()).render(vars);
        int image = c.tokenizer().vocabulary().id("<|image|>");
        int audio = c.tokenizer().vocabulary().id("<|audio|>");
        List<Integer> ids = new ArrayList<>();
        for (int id : SpecialTokens.encode(c.tokenizer(), rendered).toArray()) {
            if (id == image || id == audio) for (int r = 0; r < ROWS; r++) ids.add(-1);
            else ids.add(id);
        }
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    private static Map<String, Object> jinjaMessage(Message m) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", m.role().name());
        List<Object> parts = new ArrayList<>();
        List<Object> calls = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        boolean media = false;
        for (Content part : m.content()) {
            switch (part) {
                case Content.Text t -> {
                    text.append(t.text());
                    parts.add(Map.of("type", "text", "text", t.text()));
                }
                case Content.Media blob -> {
                    media = true;
                    parts.add(
                            Map.of(
                                    "type",
                                    blob.media() instanceof Media.Image ? "image" : "audio"));
                }
                case Content.Reasoning r -> out.put("reasoning_content", r.text());
                case Content.ToolCall call ->
                        calls.add(
                                Map.of(
                                        "function",
                                        ordered(
                                                "name",
                                                call.name(),
                                                "arguments",
                                                call.arguments())));
                case Content.ToolResult r -> text.append(r.text());
                default -> throw new AssertionError(part);
            }
        }
        out.put("content", media ? parts : text.toString());
        if (!calls.isEmpty()) out.put("tool_calls", calls);
        return out;
    }

    // --- our side
    // ---------------------------------------------------------------------------------

    private static int[] encode(
            Lfm3ChatTemplate template, Conversation conversation, int capacity) {
        List<Integer> ids = new ArrayList<>();
        template.encode(
                conversation,
                capacity,
                batch -> {
                    switch (batch.input()) {
                        case Batch.Input.Tokens t -> {
                            for (int id : t.ids()) ids.add(id);
                        }
                        case Batch.Input.Embeddings e -> {
                            for (int r = 0; r < batch.count(); r++) ids.add(-1);
                        }
                        default -> throw new AssertionError(batch);
                    }
                });
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    private static Conversation conversation(Message... messages) {
        return new Conversation(List.of(messages), List.of(), false);
    }

    private static Map<String, Object> ordered(Object... pairs) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) map.put((String) pairs[i], pairs[i + 1]);
        return map;
    }

    /** Both modalities, {@link #ROWS} zero rows each. */
    private record FakeTowers(Arena arena) implements Multimodal {
        @Override
        public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
            return Optional.of(
                    new MediaProjector<>() {
                        @Override
                        public int positions(R media) {
                            return ROWS;
                        }

                        @Override
                        public void project(
                                R media, int maxChunkSize, Consumer<MemoryView<?>> sink) {
                            sink.accept(
                                    Views.allocateF32(MemoryAllocators.ofArena(arena), ROWS, 4));
                        }
                    });
        }
    }
}
