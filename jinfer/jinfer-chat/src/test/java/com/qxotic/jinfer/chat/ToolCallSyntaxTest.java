package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
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
    void quoteThenDelimiterInsideContentStillClosesEarly() {
        // the documented ceiling: «"hi", she said» closes at the comma - the parse then fails
        // and the span is no call, exactly the pre-leniency behavior
        List<Part.ToolCall> calls = ToolCallSyntax.parseBlock("[f(a=\"hi\", she said\", b=1)]");
        assertTrue(calls.isEmpty(), String.valueOf(calls));
    }
}
