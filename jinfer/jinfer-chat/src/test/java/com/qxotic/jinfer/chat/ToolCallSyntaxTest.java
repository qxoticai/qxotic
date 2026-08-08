package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The pythonic call grammar's string leniency: models emit unescaped quotes inside quoted argument
 * values verbatim (observed on LFM2.5 despite the syntax), so a quote closes a string only when
 * what follows can continue the grammar.
 */
public final class ToolCallSyntaxTest {

    @Test
    void unescapedInnerQuotesAreContent() {
        List<Part.ToolCall> calls =
                ToolCallSyntax.parseBlock("[send_message(text=\"He said \"grüß dich\" 🌊.\")]");
        assertEquals(1, calls.size());
        assertEquals("send_message", calls.get(0).name());
        assertEquals("He said \"grüß dich\" 🌊.", calls.get(0).arguments().get("text"));
    }

    @Test
    void unescapedInnerSingleQuote() {
        List<Part.ToolCall> calls = ToolCallSyntax.parseBlock("[note(text='it's fine')]");
        assertEquals(1, calls.size());
        assertEquals("it's fine", calls.get(0).arguments().get("text"));
    }

    @Test
    void escapedQuotesStillWork() {
        List<Part.ToolCall> calls =
                ToolCallSyntax.parseBlock("[f(a=\"say \\\"hi\\\"\", b='x', n=2)]");
        assertEquals(1, calls.size());
        assertEquals("say \"hi\"", calls.get(0).arguments().get("a"));
        assertEquals("x", calls.get(0).arguments().get("b"));
        assertEquals(2L, calls.get(0).arguments().get("n"));
    }

    @Test
    void functionXmlSpan() {
        // the Qwen 3.5 / Nemotron shared form: values are typed when valid JSON, raw otherwise
        List<Part.ToolCall> calls =
                ToolCallSyntax.parseFunctionXml(
                        "\n<function=book_flight>\n<parameter=origin>\nZurich\n</parameter>\n"
                                + "<parameter=passengers>\n3\n</parameter>\n"
                                + "<parameter=flexible>\ntrue\n</parameter>\n"
                                + "<parameter=filters>\n{\"stars\": 4}\n</parameter>\n"
                                + "</function>\n");
        assertEquals(1, calls.size());
        var call = calls.get(0);
        assertEquals("book_flight", call.name());
        assertEquals("Zurich", call.arguments().get("origin"));
        assertEquals(3L, call.arguments().get("passengers"));
        assertEquals(Boolean.TRUE, call.arguments().get("flexible"));
        assertEquals(Map.of("stars", 4L), call.arguments().get("filters"));

        // a multi-line raw value spans newlines up to its closing tag
        List<Part.ToolCall> multi =
                ToolCallSyntax.parseFunctionXml(
                        "<function=send_message>\n<parameter=text>\nline one\nline two\n"
                                + "</parameter>\n</function>");
        assertEquals("line one\nline two", multi.get(0).arguments().get("text"));

        // no function element: no call
        assertTrue(ToolCallSyntax.parseFunctionXml("just text").isEmpty());
    }

    @Test
    void quoteThenDelimiterInsideContentStillClosesEarly() {
        // the documented ceiling: «"hi", she said» closes at the comma - the parse then fails
        // and the span is no call, exactly the pre-leniency behavior
        List<Part.ToolCall> calls = ToolCallSyntax.parseBlock("[f(a=\"hi\", she said\", b=1)]");
        assertTrue(calls.isEmpty(), String.valueOf(calls));
    }
}
