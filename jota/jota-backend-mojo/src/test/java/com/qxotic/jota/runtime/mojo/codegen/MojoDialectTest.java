package com.qxotic.jota.runtime.mojo.codegen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class MojoDialectTest {

    @Test
    void parsesMojoVersionOutput() {
        assertEquals(
                Optional.of(MojoDialect.V0_26_1),
                MojoDialect.fromVersionText("Mojo 0.26.1.0 (156d3ac6)"));
        assertEquals(
                Optional.of(MojoDialect.V0_26_2),
                MojoDialect.fromVersionText("Mojo 0.26.2.0 (d627decc)"));
        assertEquals(
                Optional.of(MojoDialect.V1), MojoDialect.fromVersionText("Mojo 1.2.0 (2016d0cd)"));
    }

    @Test
    void ignoresNoiseBeforeTheVersionLine() {
        String out =
                "Failed to initialize Crashpad.  Crash reporting will not be available.\n"
                        + "Mojo 1.2.0 (2016d0cd)\n";
        assertEquals(Optional.of(MojoDialect.V1), MojoDialect.fromVersionText(out));
    }

    @Test
    void parsesBareVersions() {
        assertEquals(Optional.of(MojoDialect.V0_26_1), MojoDialect.fromVersionText("0.26.1"));
        assertEquals(Optional.of(MojoDialect.V0_26_1), MojoDialect.fromVersionText("0.25.7.0"));
        assertEquals(Optional.of(MojoDialect.V0_26_2), MojoDialect.fromVersionText("0.26.3"));
        assertEquals(Optional.of(MojoDialect.V1), MojoDialect.fromVersionText("1.0"));
        assertEquals(Optional.of(MojoDialect.V0_26_1), MojoDialect.fromVersionText("25.4.0"));
    }

    @Test
    void unknownTextGivesNoDialect() {
        assertTrue(MojoDialect.fromVersionText("command not found").isEmpty());
        assertTrue(MojoDialect.fromVersionText(null).isEmpty());
    }

    @Test
    void compileFunctionCallMatchesEachRelease() {
        assertEquals(
                "ctx.compile_function[k, k]()",
                MojoDialect.V0_26_1.compileFunctionCall("ctx", "k"));
        assertEquals(
                "ctx.compile_function[k, k]()",
                MojoDialect.V0_26_2.compileFunctionCall("ctx", "k"));
        assertEquals("ctx.compile_function[k]()", MojoDialect.V1.compileFunctionCall("ctx", "k"));
    }
}
