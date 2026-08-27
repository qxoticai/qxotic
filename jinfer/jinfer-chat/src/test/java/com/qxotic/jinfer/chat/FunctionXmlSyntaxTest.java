package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * The XML function form ({@code <function=NAME><parameter=K>...</parameter></function>}) that Qwen
 * 3.5 and Nemotron emit, exercised the way a model emits it: the template's exact rendering, every
 * whitespace liberty a model takes with it, every value type, and garbage. The one law: the parser
 * never throws, because it runs inside the sampler on every token of a claimed call span.
 */
class FunctionXmlSyntaxTest {

    private static Map<String, Object> args(String span) {
        List<Content.ToolCall> calls = ToolCallSyntax.parseFunctionXml(span);
        assertEquals(1, calls.size(), span);
        return calls.get(0).arguments();
    }

    // ---- the templates' own renderings ----

    @Test
    void qwen35TemplateRendering() {
        // the span between <tool_call> and </tool_call> as Qwen 3.5's template prints it
        String span =
                "\n<function=get_weather>\n<parameter=location>\nParis, France\n</parameter>\n"
                        + "<parameter=unit>\ncelsius\n</parameter>\n</function>\n";
        List<Content.ToolCall> calls = ToolCallSyntax.parseFunctionXml(span);
        assertEquals(1, calls.size());
        assertEquals("", calls.get(0).id(), "the caller assigns ids");
        assertEquals("get_weather", calls.get(0).name());
        assertEquals(
                Map.of("location", "Paris, France", "unit", "celsius"), calls.get(0).arguments());
    }

    @Test
    void nemotronTemplateRenderingWithJsonValues() {
        String span =
                "<function=book>\n<parameter=guests>\n2\n</parameter>\n"
                        + "<parameter=dates>\n[\"2026-09-01\", \"2026-09-03\"]\n</parameter>\n"
                        + "<parameter=prefs>\n{\"smoking\": false, \"floor\": 3.5}\n</parameter>\n"
                        + "</function>";
        Map<String, Object> args = args(span);
        assertEquals(2L, args.get("guests"));
        assertEquals(List.of("2026-09-01", "2026-09-03"), args.get("dates"));
        assertEquals(Map.of("smoking", false, "floor", 3.5), args.get("prefs"));
    }

    // ---- structure ----

    @Test
    void argumentOrderIsTheEmissionOrder() {
        Map<String, Object> args =
                args(
                        "<function=f><parameter=z>\n1\n</parameter><parameter=a>\n2\n</parameter>"
                                + "<parameter=m>\n3\n</parameter></function>");
        assertEquals(List.of("z", "a", "m"), new ArrayList<>(args.keySet()));
    }

    @Test
    void nameAndKeysAreStrippedEmptyKeysSkippedDuplicatesLastWins() {
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseFunctionXml(
                        "<function= search >\n<parameter= city >\nParis\n</parameter>\n"
                                + "<parameter=>\nlost\n</parameter>\n"
                                + "<parameter=city>\nRome\n</parameter>\n</function>");
        assertEquals("search", calls.get(0).name());
        assertEquals(Map.of("city", "Rome"), calls.get(0).arguments());
    }

    @Test
    void textAroundTheFunctionIsIgnoredAndOnlyTheFirstFunctionCounts() {
        Map<String, Object> args =
                args(
                        "thinking aloud <function=f><parameter=k>\n"
                                + "v\n"
                                + "</parameter></function> trailing <function=g><parameter=x>\n"
                                + "y\n"
                                + "</parameter></function>");
        assertEquals(Map.of("k", "v"), args, "one function per span, by contract");
    }

    @Test
    void missingClosingFunctionTagStillParsesTheParameters() {
        assertEquals(Map.of("k", "v"), args("<function=f><parameter=k>\nv\n</parameter>"));
        assertEquals(Map.of(), args("<function=f>"), "a name alone is a call with no arguments");
    }

    @Test
    void noFunctionIsNoCall() {
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml(""));
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml("<parameter=k>\nv\n</parameter>"));
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml("<function=>"));
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml("<function=f"));
    }

    // ---- values ----

    @Test
    void valueTypesRoundTrip() {
        Map<String, Object> args =
                args(
                        "<function=f>"
                                + "<parameter=word>\nParis\n</parameter>"
                                + "<parameter=quoted>\n\"Paris\"\n</parameter>"
                                + "<parameter=integer>\n-42\n</parameter>"
                                + "<parameter=decimal>\n2.5\n</parameter>"
                                + "<parameter=jsonTrue>\ntrue\n</parameter>"
                                + "<parameter=pyFalse>\nFalse\n</parameter>"
                                + "<parameter=pyNone>\nNone\n</parameter>"
                                + "<parameter=jsonNull>\nnull\n</parameter>"
                                + "<parameter=brokenJson>\n{oops\n</parameter>"
                                + "<parameter=unicode>\nZürich 東京 🌍\n</parameter>"
                                + "</function>");
        assertEquals("Paris", args.get("word"));
        assertEquals("Paris", args.get("quoted"), "a JSON string is its content");
        assertEquals(-42L, args.get("integer"));
        assertEquals(2.5, args.get("decimal"));
        assertEquals(Boolean.TRUE, args.get("jsonTrue"));
        assertEquals(Boolean.FALSE, args.get("pyFalse"));
        assertTrue(args.containsKey("pyNone") && args.get("pyNone") == null);
        assertTrue(args.containsKey("jsonNull") && args.get("jsonNull") == null);
        assertEquals("{oops", args.get("brokenJson"), "invalid JSON stays the raw string");
        assertEquals("Zürich 東京 🌍", args.get("unicode"));
    }

    @Test
    void valuesAreVerbatimBetweenTheFrame() {
        Map<String, Object> args =
                args(
                        "<function=f>"
                                + "<parameter=spaced>\n  padded  \n</parameter>"
                                + "<parameter=multi>\nline one\n\nline three\n</parameter>"
                                + "<parameter=angles>\na < b > c\n</parameter>"
                                + "<parameter=tagLike>\n\"<parameter=not_a_key>\"\n</parameter>"
                                + "<parameter=after>\nok\n</parameter>"
                                + "</function>");
        assertEquals("  padded  ", args.get("spaced"), "only the frame newlines are trimmed");
        assertEquals("line one\n\nline three", args.get("multi"));
        assertEquals("a < b > c", args.get("angles"));
        assertEquals("<parameter=not_a_key>", args.get("tagLike"), "a tag inside a value is text");
        assertEquals("ok", args.get("after"), "and the real next parameter is still found");
        assertEquals(5, args.size());
    }

    @Test
    void largeValuesParseWhole() {
        String big = "x".repeat(1 << 20);
        Map<String, Object> args =
                args("<function=f><parameter=blob>\n" + big + "\n</parameter></function>");
        assertEquals(big, args.get("blob"));
    }

    // ---- resilience ----

    @Test
    void everyPrefixAndSuffixOfARealSpanParsesWithoutThrowing() {
        String span =
                "\n<function=search>\n<parameter=city>\nParis\n</parameter>\n"
                        + "<parameter=filters>\n{\"stars\": 4, \"tags\": [\"a\", \"b\"]}\n"
                        + "</parameter>\n<parameter=empty>\n</parameter>\n</function>\n";
        for (int cut = 0; cut <= span.length(); cut++) {
            String prefix = span.substring(0, cut), suffix = span.substring(cut);
            assertDoesNotThrow(() -> ToolCallSyntax.parseFunctionXml(prefix), prefix);
            assertDoesNotThrow(() -> ToolCallSyntax.parseFunctionXml(suffix), suffix);
        }
    }

    @Test
    void randomTagSoupNeverThrowsAndAlwaysYieldsAWellFormedResult() {
        String[] fragments = {
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
            ">",
            "<",
            "=",
            "\n",
            "\n\n",
            "f",
            "k",
            "v",
            " ",
            "{",
            "}",
            "\"",
            "[",
            "]",
            ",",
            "True",
            "None",
            "3"
        };
        Random rnd = new Random(5150);
        for (int i = 0; i < 20_000; i++) {
            StringBuilder sb = new StringBuilder();
            int n = rnd.nextInt(40);
            for (int j = 0; j < n; j++) sb.append(fragments[rnd.nextInt(fragments.length)]);
            String span = sb.toString();
            List<Content.ToolCall> calls =
                    assertDoesNotThrow(() -> ToolCallSyntax.parseFunctionXml(span), span);
            assertTrue(calls.size() <= 1, span);
            for (Content.ToolCall call : calls) {
                assertNotNull(call.name(), span);
                assertTrue(!call.name().isEmpty(), span);
                assertNotNull(call.arguments(), span);
                for (String key : call.arguments().keySet()) assertTrue(!key.isEmpty(), span);
            }
        }
    }
}
