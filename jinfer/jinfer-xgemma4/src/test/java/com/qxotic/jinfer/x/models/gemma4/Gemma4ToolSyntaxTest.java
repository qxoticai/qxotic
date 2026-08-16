package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The model-free parse contract for Gemma 4's compact tool-call notation. */
class Gemma4ToolSyntaxTest {

    @Test
    void parsesStringsNumbersBooleansAndNestedValues() {
        var calls =
                Gemma4ToolSyntax.parseBlock(
                        "call:search{q:<|\"|>rivers<|\"|>,top_k:3,deep:true,"
                                + "opts:{lang:<|\"|>de<|\"|>},ids:[1,2]}");
        assertEquals(1, calls.size());
        assertEquals("search", calls.getFirst().name());
        assertEquals(
                Map.of(
                        "q",
                        "rivers",
                        "top_k",
                        3L,
                        "deep",
                        true,
                        "opts",
                        Map.of("lang", "de"),
                        "ids",
                        List.of(1L, 2L)),
                calls.getFirst().arguments());
    }

    @Test
    void quotedTextMayContainStructuralCharacters() {
        var call = Gemma4ToolSyntax.parseBlock("call:note{text:<|\"|>a,b:{c}[d]<|\"|>}").getFirst();
        assertEquals("a,b:{c}[d]", call.arguments().get("text"));
    }

    @Test
    void emptyArgumentsAndWhitespaceAreAccepted() {
        assertEquals("ping", Gemma4ToolSyntax.parseBlock("call:ping{}").getFirst().name());
        assertEquals(
                Map.of("a", 1L),
                Gemma4ToolSyntax.parseBlock(" call:x{ a : 1 } ").getFirst().arguments());
    }

    @Test
    void malformedPayloadsProduceNoCalls() {
        assertTrue(Gemma4ToolSyntax.parseBlock("not a call").isEmpty());
        assertTrue(Gemma4ToolSyntax.parseBlock("call:x{unterminated:<|\"|>oops}").isEmpty());
        assertTrue(Gemma4ToolSyntax.parseBlock("call:{}").isEmpty());
    }
}
