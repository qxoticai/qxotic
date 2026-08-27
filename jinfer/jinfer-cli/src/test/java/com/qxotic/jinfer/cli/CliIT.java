package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The CLI's two modes end to end on a real checkpoint: the one-shot instruct path (raw and
 * templated), the interactive loop driven through piped stdin, and the persistent prompt cache's
 * second-run restore. Main itself is not forked (System.exit); the modes run in-process with the
 * console captured.
 */
@Tag("integration")
class CliIT {

    private static final String REF = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";

    private static Options instruct(Path model, Path cache, boolean readOnly, String... extra) {
        return new Options(
                model, Map.of(), null, "Say hi", null, false,
                0f, // greedy: the reply is deterministic
                null, null, null, 42L, 32, 512, false, // stream off: the reply lands in Turn's text
                false, true, false, false, false, cache, readOnly, 4);
    }

    private static ChatEngine engine(Options options) throws IOException {
        var model = AOT.load(options.modelPath(), options.companions(), options.tokenizerPath());
        return new ChatEngine(
                model,
                "test",
                PromptCache.Options.DEFAULTS
                        .withContextCapacity(
                                options.contextCapacity() == null ? 0 : options.contextCapacity())
                        .withCatalog(options.promptCache(), options.promptCacheReadOnly()));
    }

    @Test
    void instructAnswersAndReportsTimings(@TempDir Path dir) throws IOException {
        Path model = TestModels.require(REF);
        Options options = instruct(model, null, false);
        PrintStream realOut = System.out;
        PrintStream realErr = System.err;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        ChatEngine e = engine(options);
        try {
            Instruct.run(e, options.sampling(e.loaded().samplingDefaults()), options);
        } finally {
            e.close();
            System.setOut(realOut);
            System.setErr(realErr);
        }
        String reply = out.toString(StandardCharsets.UTF_8);
        String timings = err.toString(StandardCharsets.UTF_8);
        assertFalse(reply.isBlank(), "a greedy 32-token reply must say something");
        assertTrue(timings.contains("tokens/s"), timings);
        assertTrue(timings.contains("cache:"), timings);
    }

    @Test
    void rawPromptBypassesTheTemplate(@TempDir Path dir) throws IOException {
        Path model = TestModels.require(REF);
        Options options =
                new Options(
                        model,
                        Map.of(),
                        null,
                        "The capital of France is",
                        null,
                        false,
                        0f,
                        null,
                        null,
                        null,
                        42L,
                        16,
                        512,
                        false,
                        false,
                        true,
                        false,
                        false,
                        true,
                        null,
                        false,
                        4);
        PrintStream realErr = System.err;
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        ChatEngine e = engine(options);
        try {
            Instruct.run(e, options.sampling(e.loaded().samplingDefaults()), options);
        } finally {
            e.close();
            System.setErr(realErr);
        }
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("tokens/s"));
    }

    @Test
    void chatLoopRunsTwoTurnsAndQuits(@TempDir Path dir) throws IOException {
        Path model = TestModels.require(REF);
        Options options =
                new Options(
                        model, Map.of(), null, null, null, true, 0f, null, null, null, 42L, 16, 512,
                        false, false, true, false, false, false, null, false, 4);
        InputStream realIn = System.in;
        PrintStream realErr = System.err;
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setIn(
                new ByteArrayInputStream(
                        "Say hi\n/context\nSay bye\n/quit\n".getBytes(StandardCharsets.UTF_8)));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        ChatEngine e = engine(options);
        try {
            Chat.run(e, options.sampling(e.loaded().samplingDefaults()), options);
        } finally {
            e.close();
            System.setIn(realIn);
            System.setErr(realErr);
        }
        String timings = err.toString(StandardCharsets.UTF_8);
        // two turns completed, each with its summary line - the second served from the session
        assertEquals(2, timings.split("cache:", -1).length - 1, timings);
        assertTrue(timings.contains("cache: session"), timings);
    }

    @Test
    void aRefusedTurnEndsThatTurnNotTheChat(@TempDir Path dir) throws IOException {
        // a prompt over the context used to escape run() with a stack trace, losing the session
        Path model = TestModels.require(REF);
        Options options =
                new Options(
                        model, Map.of(), null, null, null, true, 0f, null, null, null, 42L, 16, 64,
                        false, false, true, false, false, false, null, false, 4);
        InputStream realIn = System.in;
        PrintStream realErr = System.err;
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setIn(
                new ByteArrayInputStream(
                        ("word ".repeat(300) + "\nSay hi\n/quit\n")
                                .getBytes(StandardCharsets.UTF_8)));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        ChatEngine e = engine(options);
        try {
            Chat.run(e, options.sampling(e.loaded().samplingDefaults()), options);
        } finally {
            e.close();
            System.setIn(realIn);
            System.setErr(realErr);
        }
        String output = err.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("ERROR"), output);
        assertEquals(
                1, output.split("cache:", -1).length - 1, "the next turn still ran: " + output);
    }

    @Test
    void theCacheRestoresThePromptOnTheSecondRun(@TempDir Path dir) throws IOException {
        Path model = TestModels.require(REF);
        Path cache = dir.resolve("prompts.jkv");
        Sampling sampling;
        ChatEngine first = engine(instruct(model, cache, false));
        try {
            sampling = first.loaded().samplingDefaults().resolve(0f, null, null, null, 42L);
            Instruct.run(first, sampling, instruct(model, cache, false));
        } finally {
            first.close();
        }
        assertTrue(Files.exists(cache), "--cache must leave its artifact");
        long sizeAfterFirst = Files.size(cache);

        PrintStream realErr = System.err;
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        ChatEngine e = engine(instruct(model, cache, true));
        try {
            Instruct.run(e, sampling, instruct(model, cache, true));
        } finally {
            e.close();
            System.setErr(realErr);
        }
        assertTrue(
                err.toString(StandardCharsets.UTF_8).contains("restored"),
                err.toString(StandardCharsets.UTF_8));
        assertEquals(sizeAfterFirst, Files.size(cache), "a read-only cache never grows");
    }
}
