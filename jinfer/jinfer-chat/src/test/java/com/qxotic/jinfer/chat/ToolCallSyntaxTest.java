package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The tool-call payload grammars against what small models actually emit (langchain4j's
 * ToolExecutionRequestUtil tolerance cases, moved down to where jinfer parses): trailing commas,
 * double-encoded argument strings, escaped quotes, the {@code parameters} alias, the XML function
 * form's typed values - and the line in the sand: unparseable payloads drop the call instead of
 * fabricating one.
 */
final class ToolCallSyntaxTest {

    @Test
    void cleanEnvelopeSingleAndArray() {
        List<Content.ToolCall> one =
                ToolCallSyntax.parseBlock("{\"name\":\"f\",\"arguments\":{\"x\":1}}");
        assertEquals(1, one.size());
        assertEquals("f", one.get(0).name());
        assertEquals(Map.of("x", 1L), one.get(0).arguments());

        List<Content.ToolCall> two =
                ToolCallSyntax.parseBlock(
                        "[{\"name\":\"f\",\"arguments\":{}},{\"name\":\"g\",\"arguments\":null}]");
        assertEquals(List.of("f", "g"), two.stream().map(Content.ToolCall::name).toList());
        assertEquals(Map.of(), two.get(1).arguments(), "null arguments mean no arguments");
    }

    @Test
    void parametersIsAnAliasForArguments() {
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock("{\"name\":\"f\",\"parameters\":{\"x\":1}}");
        assertEquals(Map.of("x", 1L), calls.get(0).arguments());
    }

    @Test
    void trailingCommasAreSalvaged() {
        // the langchain4j argument_comma case, one layer down: strict parse would drop this call
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock("{\"name\":\"f\",\"arguments\":{\"city\":\"Paris\",},}");
        assertEquals(1, calls.size());
        assertEquals(Map.of("city", "Paris"), calls.get(0).arguments());

        // and in Mistral's [ARGS] object body
        assertEquals(
                Map.of("x", 1L, "y", List.of(1L, 2L)),
                ToolCallSyntax.parseObject("{\"x\":1,\"y\":[1,2,],}"));
    }

    @Test
    void trailingCommaInsideAStringIsContent() {
        // the salvage is string-aware: "a,}" is a string value, never a syntax error to fix
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock("{\"name\":\"f\",\"arguments\":{\"text\":\"a,}\"}}");
        assertEquals(Map.of("text", "a,}"), calls.get(0).arguments());
    }

    @Test
    void doubleEncodedArgumentsAreUnwrapped() {
        // the langchain4j leading_trailing_quotes case: arguments as a JSON string holding JSON
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock(
                        "{\"name\":\"f\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}");
        assertEquals(Map.of("city", "Paris"), calls.get(0).arguments());
    }

    @Test
    void plainStringArgumentsLandUnderAConventionalKey() {
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock("{\"name\":\"f\",\"arguments\":\"just do it\"}");
        assertEquals(Map.of("value", "just do it"), calls.get(0).arguments());
    }

    @Test
    void escapedQuotesInArgumentsSurvive() {
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseBlock(
                        "{\"name\":\"f\",\"arguments\":{\"text\":\"he said \\\"hi\\\"\"}}");
        assertEquals(Map.of("text", "he said \"hi\""), calls.get(0).arguments());
    }

    @Test
    void garbageDropsTheCallWithoutThrowing() {
        assertEquals(List.of(), ToolCallSyntax.parseBlock("not json at all"));
        assertEquals(
                List.of(), ToolCallSyntax.parseBlock("{\"name\":\"f\",\"arguments\":{\"x\":}"));
        assertEquals(List.of(), ToolCallSyntax.parseBlock("[]")); // no envelope entries
        assertEquals(List.of(), ToolCallSyntax.parseBlock("[{\"arguments\":{}}]")); // nameless
        assertNull(ToolCallSyntax.parseObject("[1,2,3]"));
        assertNull(ToolCallSyntax.parseObject("{\"x\":}"));
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource(
            delimiter = '|',
            value = {
                "template frame      | >\\nParis\\n</ | Paris",
                "empty, one newline  | >\\n</           | ''",
                "empty, two newlines | >\\n\\n</        | ''",
                "no newline after >  | >Paris\\n</    | Paris",
                "no newline before < | >\\nParis</    | Paris",
                "bare value          | >Paris</      | Paris",
                "inner newline kept  | >\\na\\nb\\n</   | a\\nb",
            })
    void functionXmlFramingNewlinesAreOptional(String label, String frame, String expected) {
        // the templates print ">\n" + value + "\n</parameter>"; a model may drop either newline
        // (the one-newline empty value used to throw from inside the sampler)
        String span = "<function=f><parameter=k" + unescape(frame) + "parameter></function>";
        List<Content.ToolCall> calls = ToolCallSyntax.parseFunctionXml(span);
        assertEquals(1, calls.size());
        assertEquals(Map.of("k", unescape(expected)), calls.get(0).arguments(), label);
    }

    @Test
    void functionXmlNeverThrowsOnTruncatedOrOddSpans() {
        String span =
                "<function=search>\n<parameter=city>\nParis\n</parameter>\n"
                        + "<parameter=limit>\n3\n</parameter>\n</function>";
        for (int cut = 0; cut <= span.length(); cut++) {
            String prefix = span.substring(0, cut);
            assertDoesNotThrow(() -> ToolCallSyntax.parseFunctionXml(prefix), prefix);
        }
        for (String odd :
                List.of(
                        "",
                        "<function=>",
                        "<function=f>",
                        "<function=f><parameter=>",
                        "<function=f><parameter=k>",
                        "<function=f><parameter=k></parameter>",
                        "<function=f><parameter=k>v</parameter><parameter=")) {
            assertDoesNotThrow(() -> ToolCallSyntax.parseFunctionXml(odd), odd);
        }
        assertEquals(
                Map.of("k", ""),
                ToolCallSyntax.parseFunctionXml("<function=f><parameter=k></parameter>")
                        .get(0)
                        .arguments());
    }

    private static String unescape(String s) {
        return s.replace("\\n", "\n");
    }

    @Test
    void xmlFunctionFormTypesItsValues() {
        // Qwen 3.5 / Nemotron: values are tojson-for-objects, raw strings otherwise, with the
        // Python spellings typed back
        List<Content.ToolCall> calls =
                ToolCallSyntax.parseFunctionXml(
                        "<function=search>\n<parameter=city>\nParis\n</parameter>\n"
                                + "<parameter=limit>\n3\n</parameter>\n"
                                + "<parameter=filters>\n{\"stars\": 4}\n</parameter>\n"
                                + "<parameter=recursive>\nTrue\n</parameter>\n"
                                + "<parameter=note>\nNone\n</parameter>\n</function>");
        assertEquals(1, calls.size());
        assertEquals("search", calls.get(0).name());
        Map<String, Object> args = calls.get(0).arguments();
        assertEquals("Paris", args.get("city"), "a bare word is a string, not broken JSON");
        assertEquals(3L, args.get("limit"));
        assertEquals(Map.of("stars", 4L), args.get("filters"));
        assertEquals(Boolean.TRUE, args.get("recursive"));
        assertTrue(args.containsKey("note") && args.get("note") == null, "None types back to null");
    }

    @Test
    void xmlFunctionFormRejectsTheMalformed() {
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml("no function here"));
        assertEquals(List.of(), ToolCallSyntax.parseFunctionXml("<function=>x</function>"));
    }
}
