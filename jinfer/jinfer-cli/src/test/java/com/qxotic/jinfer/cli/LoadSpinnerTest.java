package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class LoadSpinnerTest {
    @Test
    void redirectedOutputKeepsOnePlainLine() {
        var capture = new CliFixtures.Capture("");
        try (var spinner = LoadSpinner.start("Loading model", capture.io)) {
            assertEquals("Loading model ...\n", capture.err().replace("\r\n", "\n"));
        }
        assertEquals("Loading model ...\n", capture.err().replace("\r\n", "\n"));
        assertEquals("", capture.out());
    }

    @Test
    void dotsAppendAndClosingLeavesTheLineOnce() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var frames = new CountDownLatch(2);
        var out =
                new PrintStream(bytes, true, StandardCharsets.UTF_8) {
                    @Override
                    public void print(char value) {
                        super.print(value);
                        if (value == '.') frames.countDown();
                    }
                };
        var spinner = LoadSpinner.start("Loading model", out, true);
        try (spinner) {
            assertTrue(bytes.toString(StandardCharsets.UTF_8).startsWith("Loading model ..."));
            assertTrue(frames.await(5, TimeUnit.SECONDS), "loading dots did not advance");
        }
        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.replace("\r\n", "\n").matches("Loading model \\.{5,}\\n"), output);
        spinner.close();
        assertEquals(output, bytes.toString(StandardCharsets.UTF_8));
    }
}
