package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Part;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The parse side of Gemma 4's compact call notation (model-free). */
public final class Gemma4ToolSyntaxTest {

    @Test
    void parsesStringArgs() {
        List<Part.ToolCall> calls =
                Gemma4ToolSyntax.parseBlock("call:get_weather{city:<|\"|>Paris<|\"|>}");
        assertEquals(1, calls.size());
        assertEquals("get_weather", calls.get(0).name());
        assertEquals(Map.of("city", "Paris"), calls.get(0).arguments());
    }

    @Test
    void parsesNumbersBoolsNested() {
        List<Part.ToolCall> calls =
                Gemma4ToolSyntax.parseBlock(
                        "call:search{q:<|\"|>rivers<|\"|>,top_k:3,deep:true,"
                                + "opts:{lang:<|\"|>de<|\"|>},ids:[1,2]}");
        assertEquals(1, calls.size());
        Map<String, Object> args = calls.get(0).arguments();
        assertEquals("rivers", args.get("q"));
        assertEquals(3L, args.get("top_k"));
        assertEquals(Boolean.TRUE, args.get("deep"));
        assertEquals(Map.of("lang", "de"), args.get("opts"));
        assertEquals(List.of(1L, 2L), args.get("ids"));
    }

    @Test
    void quotedTextMayContainStructuralChars() {
        List<Part.ToolCall> calls =
                Gemma4ToolSyntax.parseBlock("call:note{text:<|\"|>a,b:{c}[d]<|\"|>}");
        assertEquals("a,b:{c}[d]", calls.get(0).arguments().get("text"));
    }

    @Test
    void emptyArgsAndWhitespaceTolerance() {
        assertEquals("ping", Gemma4ToolSyntax.parseBlock("call:ping{}").get(0).name());
        assertEquals(
                Map.of("a", 1L),
                Gemma4ToolSyntax.parseBlock(" call:x{ a : 1 } ").get(0).arguments());
    }

    @Test
    void malformedParsesToNoCalls() {
        assertTrue(Gemma4ToolSyntax.parseBlock("not a call").isEmpty());
        assertTrue(Gemma4ToolSyntax.parseBlock("call:x{unterminated:<|\"|>oops}").isEmpty());
        assertTrue(Gemma4ToolSyntax.parseBlock("call:{}").isEmpty());
    }
}
