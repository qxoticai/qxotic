package com.qxotic.jinfer.models.gptoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

/** Harmony channel routing, call parsing, and forced-call constraints without model weights. */
class GptOssChatTemplateTest {

    private static final String[] SPECIALS = {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>"
    };
    private static final Tokenizer TOKENIZER = new AsciiTokenizer();
    private static final GptOssChatTemplate TEMPLATE = new GptOssChatTemplate(TOKENIZER);
    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    Map.of(
                            "parameters",
                            Map.of(
                                    "type",
                                    "object",
                                    "properties",
                                    Map.of("city", Map.of("type", "string")),
                                    "required",
                                    List.of("city"))));
    private static final Tool REFRESH = new Tool("refresh_cache", Map.of("parameters", Map.of()));

    @Test
    void analysisAndFinalRouteWithoutLeakingHeaders() {
        Message message =
                parse(
                        "<|channel|>",
                        "analysis",
                        "<|message|>",
                        "thinking...",
                        "<|end|>",
                        "<|start|>",
                        "assistant",
                        "<|channel|>",
                        "final",
                        "<|message|>",
                        "The answer is 4.",
                        "<|return|>");
        Content.Reasoning reasoning =
                assertInstanceOf(Content.Reasoning.class, message.content().getFirst());
        assertEquals("thinking...", reasoning.text());
        assertEquals("The answer is 4.", message.text());
        assertEquals(
                TOKENIZER.encode("The answer is 4.").toList(),
                ((Content.Text) message.content().get(1)).verbatim().toList());
        String output = message.content().toString();
        assertTrue(!output.contains("assistant") && !output.contains("analysis"));
    }

    @Test
    void commentaryCallKeepsArgumentsAndVerbatimPayload() {
        String arguments = "{\"city\": \"Paris\"}";
        Message message =
                parse(
                        "<|channel|>",
                        "commentary to=functions.get_weather ",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        arguments,
                        "<|call|>");
        Content.ToolCall call =
                assertInstanceOf(Content.ToolCall.class, message.content().getFirst());
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
        assertEquals(TOKENIZER.encode(arguments).toList(), call.verbatim().toList());
    }

    @Test
    void malformedAndNonFunctionCallsDropWithoutEndingTheReply() {
        Message malformed =
                parse(
                        "<|channel|>",
                        "commentary to=functions.get_weather ",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        "{broken",
                        "<|end|>",
                        "<|channel|>",
                        "final",
                        "<|message|>",
                        "Still alive.",
                        "<|return|>");
        assertEquals("Still alive.", malformed.text());
        assertTrue(malformed.content().stream().noneMatch(Content.ToolCall.class::isInstance));

        Message nonFunction =
                parse(
                        "<|channel|>",
                        "commentary to=browser.search ",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        "{\"q\":\"weather\"}",
                        "<|end|>",
                        "<|channel|>",
                        "final",
                        "<|message|>",
                        "Here is the weather.",
                        "<|return|>");
        assertEquals("Here is the weather.", nonFunction.text());
        assertTrue(nonFunction.content().stream().noneMatch(Content.ToolCall.class::isInstance));
    }

    @Test
    void commentaryAndConstrainedFinalRemainContent() {
        Message message =
                parse(
                        "<|channel|>",
                        "commentary",
                        "<|message|>",
                        "Fetching.",
                        "<|end|>",
                        "<|channel|>",
                        "final ",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        "{\"answer\":42}",
                        "<|return|>");
        assertEquals("Fetching.{\"answer\":42}", message.text());
    }

    @Test
    void autoCallAcceptsAnUnofferedNameAndOptionalSpace() {
        Message unoffered =
                parse(
                        "<|channel|>",
                        "commentary to=functions.unoffered ",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        "{}",
                        "<|call|>");
        assertEquals("unoffered", ((Content.ToolCall) unoffered.content().getFirst()).name());

        Message tight =
                parse(
                        "<|channel|>",
                        "commentary to=functions.get_time",
                        "<|constrain|>",
                        "json",
                        "<|message|>",
                        "{}",
                        "<|call|>");
        assertEquals("get_time", ((Content.ToolCall) tight.content().getFirst()).name());
    }

    @Test
    void forcedCallAdmitsItsSchemaAndParsesTheResult() {
        ReplyLanguage.Selection selection = TEMPLATE.forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, "{\"city\": \"Paris\"}");
        assertTrue(walk.accepted());
        walk.feed(special("<|call|>"));
        Content.ToolCall call = (Content.ToolCall) walk.finish().content().getFirst();
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
    }

    @Test
    void forcedCallOffersOnlyDeclaredNamesAndArgumentKeys() {
        ReplyLanguage.Selection selection =
                TEMPLATE.forcedCall(List.of(WEATHER, REFRESH)).orElseThrow();
        ReplyLanguage.Walk names = selection.walk();
        names.feed(special("<|channel|>"));
        feed(names, TOKENIZER.encode("commentary to=functions.").toArray());
        MemoryView<MemorySegment> logits = zeros();
        assertTrue(names.maskLogits(logits));
        assertEquals(0f, logit(logits, 'g'));
        assertEquals(0f, logit(logits, 'r'));
        assertEquals(Float.NEGATIVE_INFINITY, logit(logits, 'x'));

        ReplyLanguage.Selection weather = TEMPLATE.forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk arguments = weather.walk();
        feed(arguments, weather.forcedPrefix());
        feed(arguments, TOKENIZER.encode("{\"").toArray());
        logits = zeros();
        assertTrue(arguments.maskLogits(logits));
        assertEquals(0f, logit(logits, 'c'));
        assertEquals(Float.NEGATIVE_INFINITY, logit(logits, 't'));
    }

    @Test
    void noParameterToolAcceptsOnlyAnEmptyObject() {
        ReplyLanguage.Selection selection = TEMPLATE.forcedCall(List.of(REFRESH)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, "{}");
        assertTrue(walk.accepted());
    }

    private static Message parse(String... pieces) {
        IntSequence.Builder tokens = IntSequence.newBuilder();
        for (String piece : pieces) {
            int special = specialOrMinusOne(piece);
            if (special >= 0) tokens.add(special);
            else TOKENIZER.encode(piece).forEachInt(tokens::add);
        }
        return ReplyParser.parse(TEMPLATE.parser(TOKENIZER), tokens.build());
    }

    private static void assertAdmitted(ReplyLanguage.Walk walk, String text) {
        for (int token : TOKENIZER.encode(text).toArray()) {
            MemoryView<MemorySegment> logits = zeros();
            assertTrue(walk.maskLogits(logits));
            assertEquals(0f, Views.getFloat(logits, token, "logits"));
            walk.feed(token);
        }
    }

    private static MemoryView<MemorySegment> zeros() {
        return Views.fromFloatArray(
                MemoryAllocators.ofArena(Arena.ofAuto()), new float[TOKENIZER.vocabulary().size()]);
    }

    private static float logit(MemoryView<MemorySegment> logits, char character) {
        return Views.getFloat(logits, AsciiTokenizer.id(character), "logits");
    }

    private static void feed(ReplyLanguage.Walk walk, int[] tokens) {
        for (int token : tokens) walk.feed(token);
    }

    private static int special(String spelling) {
        int id = specialOrMinusOne(spelling);
        if (id < 0) throw new NoSuchElementException(spelling);
        return id;
    }

    private static int specialOrMinusOne(String spelling) {
        for (int i = 0; i < SPECIALS.length; i++) if (SPECIALS[i].equals(spelling)) return i;
        return -1;
    }

    private static final class AsciiTokenizer implements Tokenizer {
        private final Vocabulary vocabulary = new AsciiVocabulary();

        static int id(char character) {
            return SPECIALS.length + character;
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++) {
                char character = text.charAt(i);
                if (character >= 128) throw new IllegalArgumentException("non-ASCII test input");
                out.add(id(character));
            }
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            int token = tokens.intAt(tokenStartIndex);
            String text =
                    token < SPECIALS.length
                            ? SPECIALS[token]
                            : String.valueOf((char) (token - SPECIALS.length));
            out.put(text.getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class AsciiVocabulary implements Vocabulary {
        @Override
        public int size() {
            return SPECIALS.length + 128;
        }

        @Override
        public String token(int id) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            return id < SPECIALS.length
                    ? SPECIALS[id]
                    : String.valueOf((char) (id - SPECIALS.length));
        }

        @Override
        public int id(String text) {
            int special = specialOrMinusOne(text);
            if (special >= 0) return special;
            if (text.length() == 1 && text.charAt(0) < 128)
                return AsciiTokenizer.id(text.charAt(0));
            throw new NoSuchElementException(text);
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < size();
        }

        @Override
        public boolean contains(String text) {
            return specialOrMinusOne(text) >= 0 || (text.length() == 1 && text.charAt(0) < 128);
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            if (type == StandardTokenType.CONTROL) return id < SPECIALS.length;
            if (type == StandardTokenType.NORMAL) return id >= SPECIALS.length;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return IntStream.range(0, size()).mapToObj(id -> Map.entry(token(id), id)).iterator();
        }
    }
}
