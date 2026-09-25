package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.toknroll.IntSequence;
import org.junit.jupiter.api.Test;

class InstructTest {

    @Test
    void stdinSystemPromptAndTokenBudgetReachTheRealEngine() throws Exception {
        for (String streaming : new String[] {"--stream", "--no-stream"}) {
            var template = new CliFixtures.Template();
            var capture = new CliFixtures.Capture("A prompt from stdin.");
            Options o =
                    Options.parse(
                            "instruct",
                            "-m",
                            "unused",
                            "-",
                            "--system-prompt",
                            "Be concise.",
                            "--temp",
                            "0",
                            "-n",
                            "4",
                            streaming);
            try (var engine = CliFixtures.engine(template)) {
                Instruct.run(
                        engine,
                        o.sampling(engine.loaded().samplingDefaults()),
                        o,
                        capture.io,
                        capture.io.text(o.input));
            }
            var messages = template.conversations.getFirst().messages();
            assertEquals("Be concise.", messages.getFirst().text());
            assertEquals("A prompt from stdin.", messages.getLast().text());
            assertEquals("xxxx\n", capture.out().replace("\r\n", "\n"));
            assertFalse(capture.inputClosed);
            assertTrue(capture.err().contains("tokens/s"));
        }
    }

    @Test
    void rawInputBypassesTheTemplate() throws Exception {
        var template = new CliFixtures.Template();
        var capture = new CliFixtures.Capture("");
        Options o =
                Options.parse(
                        "instruct",
                        "-m",
                        "unused",
                        "Once upon a time",
                        "--raw-prompt",
                        "--temp",
                        "0",
                        "-n",
                        "3");
        try (var engine = CliFixtures.engine(template)) {
            Instruct.run(
                    engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io, o.input);
        }
        assertTrue(template.conversations.isEmpty());
        assertEquals("xxx\n", capture.out().replace("\r\n", "\n"));
    }

    @Test
    void anAlreadyReadDashIsTextNotAnotherStdinRead() throws Exception {
        var template = new CliFixtures.Template();
        var capture = new CliFixtures.Capture("");
        Options o = Options.parse("instruct", "-m", "unused", "-", "--temp", "0", "-n", "1");
        try (var engine = CliFixtures.engine(template)) {
            Instruct.run(
                    engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io, "-");
        }
        assertEquals("-", template.conversations.getFirst().messages().getLast().text());
    }

    @Test
    void aRawPromptStartsWithTheModelsStartTokensUnlessItAlreadyDoes() {
        int[] prompt = {7, 8, 9};
        assertArrayEquals(
                new int[] {1, 7, 8, 9}, Instruct.withPromptStart(IntSequence.of(1), prompt));
        assertSame(prompt, Instruct.withPromptStart(IntSequence.of(7), prompt), "already spelled");
        assertSame(prompt, Instruct.withPromptStart(IntSequence.empty(), prompt), "no start token");
        assertArrayEquals(
                new int[] {1, 2, 7, 8, 9}, Instruct.withPromptStart(IntSequence.of(1, 2), prompt));
        assertArrayEquals(new int[] {1}, Instruct.withPromptStart(IntSequence.of(1), new int[0]));
    }

    @Test
    void oversizedRawInputNamesTheBudgetAndItsRemedy() throws Exception {
        var capture = new CliFixtures.Capture("");
        Options o = Options.parse("instruct", "-m", "unused", "hi", "--raw-prompt", "--temp", "0");
        try (var engine = CliFixtures.engine(new CliFixtures.Template())) {
            var error =
                    assertThrows(
                            java.io.IOException.class,
                            () ->
                                    Instruct.run(
                                            engine,
                                            o.sampling(engine.loaded().samplingDefaults()),
                                            o,
                                            capture.io,
                                            "x".repeat(5000)));
            assertTrue(error.getMessage().contains("5000"));
            assertTrue(error.getMessage().contains("4096"));
            assertTrue(error.getMessage().contains("--context-capacity"));
            assertEquals("", capture.out());
        }
    }
}
