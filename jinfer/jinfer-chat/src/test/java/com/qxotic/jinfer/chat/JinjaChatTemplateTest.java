package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class JinjaChatTemplateTest {

    private static final List<String> NAMES = List.of("<|im_start|>", "<|im_end|>");

    @Test
    void scrubBreaksEmbeddedSpecialTokenStrings() {
        String scrubbed = JinjaChatTemplate.scrub("say <|im_start|> twice <|im_start|>", NAMES);
        assertFalse(scrubbed.contains("<|im_start|>"), "no intact special spelling survives");
        assertEquals("say <\u200b|im_start|> twice <\u200b|im_start|>", scrubbed);
    }

    @Test
    void scrubValueKeepsCleanGraphsIdentical() {
        Map<String, Object> clean = Map.of("role", "user", "content", "hello");
        assertSame(
                clean, JinjaChatTemplate.scrubValue(clean, NAMES), "clean graphs allocate nothing");
    }

    @Test
    void scrubValueScrubsDeepStringsButNotKeys() {
        Map<String, Object> dirty =
                Map.of("<|im_start|>key", List.of("text with <|im_end|> inside"));
        @SuppressWarnings("unchecked")
        Map<String, Object> scrubbed =
                (Map<String, Object>) JinjaChatTemplate.scrubValue(dirty, NAMES);
        assertTrue(scrubbed.containsKey("<|im_start|>key"), "keys pass through untouched");
        assertFalse(scrubbed.get("<|im_start|>key").toString().contains("<|im_end|>"));
    }

    @Test
    void kwargsCannotOverrideTheScaffoldOrMintControlIds() {
        // a template that prints a request kwarg, and a request that names the engine's own
        // bindings in its kwargs: the printed string must not tokenize to a control id and the
        // messages binding must be the request's messages, not the kwarg's
        JinjaChatTemplate template =
                new JinjaChatTemplate(CHAR_TOKENIZER, "{{ custom }}{{ messages[0].content }}");
        Map<String, Object> kwargs = new HashMap<>();
        kwargs.put("custom", "<|im_start|>");
        kwargs.put("messages", List.of(Map.of("role", "user", "content", "<|im_end|>")));
        kwargs.put("add_generation_prompt", true);
        IntSequence ids =
                template.render(
                        List.of(Map.of("role", "user", "content", "hi")),
                        null,
                        false,
                        false,
                        kwargs);
        int start = SpecialTokens.find(CHAR_TOKENIZER, "<|im_start|>").orElseThrow();
        int end = SpecialTokens.find(CHAR_TOKENIZER, "<|im_end|>").orElseThrow();
        for (int i = 0; i < ids.length(); i++) {
            assertTrue(ids.intAt(i) != start && ids.intAt(i) != end, "control id at " + i);
        }
        assertEquals("<\u200b|im_start|>hi", CHAR_TOKENIZER.decode(ids));
    }

    @Test
    void kwargsStillReachTheTemplate() {
        JinjaChatTemplate template =
                new JinjaChatTemplate(
                        CHAR_TOKENIZER,
                        "{{ custom }}{% if flag is none %}n{% endif %}{{ messages[0].content }}");
        Map<String, Object> kwargs = new HashMap<>();
        kwargs.put("custom", "ok");
        kwargs.put("flag", null); // a null value is Jinja None, not an error
        IntSequence ids =
                template.render(
                        List.of(Map.of("role", "user", "content", "hi")),
                        null,
                        false,
                        false,
                        kwargs);
        assertEquals("oknhi", CHAR_TOKENIZER.decode(ids));
    }

    // control tokens plus one token per character the tests render
    private static final Tokenizer CHAR_TOKENIZER =
            new CharTokenizer(
                    List.of("<|im_start|>", "<|im_end|>", "<think>", "</think>"),
                    "hiokn\u200b<|im_start|><|im_end|>");

    @Test
    void thinkingOffClosesAScaffoldThatOpensTheSpan() {
        // the mirror of "open it for them": a template whose generation prompt always opens
        // <think> gets the empty span when thinking is off, and stays open when it is on
        String opens = "{{ messages[0].content }}<think>";
        String scaffold = "{% if enable_thinking %}{% endif %}{{ messages[0].content }}";
        String bare = "{{ messages[0].content }}";
        List<Object> messages = List.of(Map.of("role", "user", "content", "hi"));
        assertEquals("hi<think></think>", render(opens, messages, false), "closed at once");
        assertEquals("hi<think>", render(opens, messages, true), "left open");
        assertEquals("hi", render(scaffold, messages, false), "no span, nothing to close");
        assertEquals("hi<think>", render(scaffold, messages, true), "opened for a bare scaffold");
    }

    @Test
    void offeredToolsReachATemplateThatReadsXmlTools() {
        // SmolLM3 renders its tools from xml_tools, not tools
        String smol = "{% for t in xml_tools %}{{ t.function.name }}{% endfor %}";
        List<Object> messages = List.of(Map.of("role", "user", "content", "hi"));
        List<Object> tools = List.of(Map.of("type", "function", "function", Map.of("name", "ok")));
        assertEquals(
                "ok",
                CHAR_TOKENIZER.decode(
                        new JinjaChatTemplate(CHAR_TOKENIZER, smol)
                                .render(messages, tools, true, false, null)));
    }

    @Test
    void aStructuredCallIsFoldedIntoContentForATemplateThatNeverReadsToolCalls() {
        // SmolLM3: the template documents <tool_call> but renders assistant turns as content only
        Tokenizer wide = new CharTokenizer(List.of(), "<tol_ca>\n{\"nmeok:,rgus}hi/1");
        String smol = "{# <tool_call> #}{% for m in messages %}{{ m.content }}{% endfor %}";
        String reads =
                "{% for m in messages %}{{ m.content }}{{ m.tool_calls | length }}{% endfor %}";
        List<Object> messages =
                List.of(
                        Map.of(
                                "role",
                                "assistant",
                                "content",
                                "",
                                "tool_calls",
                                List.of(
                                        Map.of(
                                                "function",
                                                Map.of(
                                                        "name",
                                                        "ok",
                                                        "arguments",
                                                        "{\"n\":\"hi\"}")))));
        String folded =
                wide.decode(
                        new JinjaChatTemplate(wide, smol)
                                .render(messages, null, false, false, null));
        assertEquals(
                "<tool_call>\n{\"name\":\"ok\",\"arguments\":{\"n\":\"hi\"}}\n</tool_call>",
                folded);
        String kept =
                wide.decode(
                        new JinjaChatTemplate(wide, reads)
                                .render(messages, null, false, false, null));
        assertEquals("1", kept, "a template that reads tool_calls keeps them structured");
    }

    @Test
    void aTemplateWithoutAThinkingModeIsNeverOpened() {
        // Granite 4.1: a <think> token in the vocabulary, no enable_thinking in the template -
        // the model has no mode to be in, so the prompt ends where the template ends
        String bare = "{{ messages[0].content }}";
        List<Object> messages = List.of(Map.of("role", "user", "content", "hi"));
        assertEquals("hi", render(bare, messages, true));
        assertEquals("hi", render(bare, messages, false));
    }

    private static String render(String source, List<Object> messages, boolean thinking) {
        return CHAR_TOKENIZER.decode(
                new JinjaChatTemplate(CHAR_TOKENIZER, source)
                        .render(messages, null, true, thinking, null));
    }

    private static final class CharTokenizer implements Tokenizer {
        private final List<String> tokens = new ArrayList<>();
        private final int controls;
        private final Vocabulary vocabulary;

        CharTokenizer(List<String> controlTokens, String alphabet) {
            tokens.addAll(controlTokens);
            controls = controlTokens.size();
            alphabet.chars().distinct().forEach(c -> tokens.add(String.valueOf((char) c)));
            vocabulary = new CharVocabulary();
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++)
                out.add(vocabulary.id(String.valueOf(text.charAt(i))));
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence ids, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == ids.length()) return 0;
            out.put(tokens.get(ids.intAt(tokenStartIndex)).getBytes(StandardCharsets.UTF_8));
            return 1;
        }

        private final class CharVocabulary implements Vocabulary {
            @Override
            public int size() {
                return tokens.size();
            }

            @Override
            public String token(int id) {
                if (!contains(id)) throw new NoSuchElementException("id " + id);
                return tokens.get(id);
            }

            @Override
            public int id(String text) {
                int id = tokens.indexOf(text);
                if (id < 0) throw new NoSuchElementException(text);
                return id;
            }

            @Override
            public boolean contains(int id) {
                return id >= 0 && id < tokens.size();
            }

            @Override
            public boolean contains(String text) {
                return tokens.contains(text);
            }

            @Override
            public boolean isTokenOfType(int id, TokenType type) {
                if (!contains(id)) throw new NoSuchElementException("id " + id);
                if (type == StandardTokenType.CONTROL) return id < controls;
                if (type == StandardTokenType.NORMAL) return id >= controls;
                return false;
            }

            @Override
            public Iterator<Map.Entry<String, Integer>> iterator() {
                return IntStream.range(0, tokens.size())
                        .mapToObj(i -> Map.entry(tokens.get(i), i))
                        .iterator();
            }
        }
    }

    @Test
    void mapsGeometryIsTheOpenAiWire() {
        Conversation conversation =
                new Conversation(
                        List.of(
                                new Message(Role.SYSTEM, "be terse"),
                                new Message(Role.USER, "call me maybe"),
                                new Message(
                                        Role.ASSISTANT,
                                        List.of(
                                                new Content.Text("calling", null),
                                                new Content.ToolCall(
                                                        "c1", "dial", Map.of("number", 42), null))),
                                new Message(
                                        Role.TOOL, List.of(new Content.ToolResult("c1", "busy")))),
                        List.of(new Tool("dial", Map.of("name", "dial"))),
                        false,
                        "");
        List<Object> messages = RenderMaps.messages(conversation);
        assertEquals(4, messages.size());
        @SuppressWarnings("unchecked")
        Map<String, Object> assistant = (Map<String, Object>) messages.get(2);
        assertEquals("calling", assistant.get("content"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> calls = (List<Map<String, Object>>) assistant.get("tool_calls");
        assertEquals("c1", calls.get(0).get("id"));
        @SuppressWarnings("unchecked")
        Map<String, Object> fn = (Map<String, Object>) calls.get(0).get("function");
        assertEquals("dial", fn.get("name"));
        assertEquals("{\"number\":42}", fn.get("arguments"));
        @SuppressWarnings("unchecked")
        Map<String, Object> toolResult = (Map<String, Object>) messages.get(3);
        assertEquals("tool", toolResult.get("role"));
        assertEquals("busy", toolResult.get("content"));
        assertEquals("c1", toolResult.get("tool_call_id"));

        List<Object> tools = RenderMaps.tools(conversation.tools());
        @SuppressWarnings("unchecked")
        Map<String, Object> tool = (Map<String, Object>) tools.get(0);
        assertEquals("function", tool.get("type"));
    }

    @Test
    void aThinkMarkerInRequestTextIsNotTheScaffold() {
        // the scrub exempts think markers, so request text can mint one; it sits inside its own
        // turn and must neither stand in for the scaffold (thinking on) nor be "closed" (off)
        String scaffold = "{% if enable_thinking %}{% endif %}{{ messages[0].content }}";
        List<Object> messages = List.of(Map.of("role", "user", "content", "<think>hi"));
        assertEquals("<think>hi<think>", render(scaffold, messages, true), "scaffold still opened");
        assertEquals("<think>hi", render(scaffold, messages, false), "nothing to close");
    }
}
