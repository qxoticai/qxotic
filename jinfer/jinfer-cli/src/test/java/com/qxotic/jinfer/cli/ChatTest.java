package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ChatTest {
    @Test
    void conversationOwnsHistoryAndRecoversFromARejectedTurn() throws Exception {
        var template = new CliFixtures.Template();
        var capture = new CliFixtures.Capture("\nfirst\n/context\nreject\nsecond\n/quit\nnever\n");
        Options o =
                Options.parse(
                        "chat",
                        "-m",
                        "unused",
                        "--system-prompt",
                        "Be concise.",
                        "--temp",
                        "0",
                        "-n",
                        "4");
        try (var engine = CliFixtures.engine(template)) {
            Chat.run(engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io);
        }
        assertEquals(2, template.conversations.size());
        assertEquals(2, template.conversations.getFirst().messages().size());
        var second = template.conversations.getLast().messages();
        assertEquals(4, second.size());
        assertEquals("Be concise.", second.getFirst().text());
        assertEquals("xxxx", second.get(2).text());
        assertEquals("second", second.getLast().text());
        assertEquals("xxxx\nxxxx\n", capture.out().replace("\r\n", "\n"));
        assertTrue(capture.err().contains("rejected test turn"));
        assertTrue(capture.err().contains("context:"));
        assertFalse(capture.inputClosed);
    }

    @Test
    void eofAndExitDoNotGenerate() throws Exception {
        for (String input : new String[] {"", "/exit\n"}) {
            var template = new CliFixtures.Template();
            var capture = new CliFixtures.Capture(input);
            Options o = Options.parse("chat", "-m", "unused");
            try (var engine = CliFixtures.engine(template)) {
                Chat.run(engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io);
            }
            assertTrue(template.conversations.isEmpty());
            assertEquals("", capture.out());
            assertFalse(capture.inputClosed);
        }
    }

    @Test
    void aPipedFinalLineNeedsNoTrailingNewline() throws Exception {
        var template = new CliFixtures.Template();
        var capture = new CliFixtures.Capture("Hello");
        Options o = Options.parse("chat", "-m", "unused", "--temp", "0", "-n", "1");
        try (var engine = CliFixtures.engine(template)) {
            Chat.run(engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io);
        }
        assertEquals("Hello", template.conversations.getFirst().messages().getFirst().text());
        assertEquals("x\n", capture.out().replace("\r\n", "\n"));
        assertFalse(capture.inputClosed);
    }

    @Test
    void anOversizedPromptIsRecoverableWithoutCatchingGenerationBugs() throws Exception {
        var template = new CliFixtures.Template();
        var capture = new CliFixtures.Capture("x".repeat(5000) + "\nhello\n/exit\n");
        Options o = Options.parse("chat", "-m", "unused", "--temp", "0", "-n", "1");
        try (var engine = CliFixtures.engine(template)) {
            Chat.run(engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io);
        }
        assertEquals("x\n", capture.out().replace("\r\n", "\n"));
        assertTrue(capture.err().contains("--context-capacity"));
        assertFalse(capture.err().contains("\tat "));
        assertEquals(
                1,
                template.conversations.getLast().messages().size(),
                "rejected input was removed from history");
    }
}
