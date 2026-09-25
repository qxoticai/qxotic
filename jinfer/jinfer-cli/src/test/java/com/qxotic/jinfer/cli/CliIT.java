package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Real model loading and generation through the new commands and the compatibility syntax. */
@Tag("integration")
class CliIT {
    private static final String REF = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";

    @Test
    void instructAndItsAliasProduceTheSameGreedyReply(@TempDir Path dir) {
        String model = TestModels.require(REF).toString();
        var modern = new CliFixtures.Capture("");
        var alias = new CliFixtures.Capture("");
        var legacy = new CliFixtures.Capture("");
        var store = ModelStore.of(dir);
        assertEquals(
                0,
                Main.run(
                        new String[] {
                            "instruct",
                            "-m",
                            model,
                            "Say hi",
                            "--temp",
                            "0",
                            "-n",
                            "16",
                            "--no-stream"
                        },
                        modern.io,
                        store),
                modern.err());
        assertEquals(
                0,
                Main.run(
                        new String[] {"-m", model, "--temp", "0", "prompt", "Say hi", "-n", "16"},
                        alias.io,
                        store),
                alias.err());
        assertEquals(
                0,
                Main.run(
                        new String[] {"-m", model, "--prompt", "Say hi", "--temp", "0", "-n", "16"},
                        legacy.io,
                        store),
                legacy.err());
        assertFalse(modern.out().isBlank());
        assertEquals(modern.out(), alias.out());
        assertEquals(modern.out(), legacy.out());
        assertTrue(modern.err().contains("tokens/s"));
    }

    @Test
    void rawPromptBypassesTheTemplate(@TempDir Path dir) {
        var capture = new CliFixtures.Capture("");
        assertEquals(
                0,
                Main.run(
                        new String[] {
                            "instruct",
                            "-m",
                            TestModels.require(REF).toString(),
                            "The capital of France is",
                            "--raw-prompt",
                            "--temp",
                            "0",
                            "-n",
                            "16"
                        },
                        capture.io,
                        ModelStore.of(dir)),
                capture.err());
        assertTrue(capture.err().contains("tokens/s"));
    }

    @Test
    void compatibleTokenizerOverrideLoadsAndGenerates(@TempDir Path dir) {
        Path model = TestModels.require(REF);
        var capture = new CliFixtures.Capture("");
        assertEquals(
                0,
                Main.run(
                        new String[] {
                            "instruct",
                            "-m",
                            model.toString(),
                            "Say hi",
                            "--with",
                            "tokenizer=" + model,
                            "--temp",
                            "0",
                            "-n",
                            "4"
                        },
                        capture.io,
                        ModelStore.of(dir)),
                capture.err());
        assertFalse(capture.out().isBlank());
        assertTrue(capture.err().contains("tokens/s"));
    }

    @Test
    void chatRunsTwoTurnsAndRecoversFromARejectedTurn() throws Exception {
        Options o =
                Options.parse(
                        "chat",
                        "-m",
                        TestModels.require(REF).toString(),
                        "--temp",
                        "0",
                        "-n",
                        "16",
                        "-c",
                        "64");
        var capture =
                new CliFixtures.Capture(
                        "word ".repeat(300) + "\nSay hi\n/context\nSay bye\n/quit\n");
        try (Arena weights = Arena.ofShared();
                var engine = engine(o, weights)) {
            Chat.run(engine, o.sampling(engine.loaded().samplingDefaults()), o, capture.io);
        }
        assertEquals(2, capture.err().split("cache:", -1).length - 1, capture.err());
        assertTrue(capture.err().contains("jinfer chat:"));
        assertTrue(capture.err().contains("cache: session"));
        assertFalse(capture.out().contains("> "));
    }

    @Test
    void cacheRestoresThePromptWithoutChangingAReadOnlyCatalog(@TempDir Path dir) throws Exception {
        Path cache = dir.resolve("prompts.jkv");
        String model = TestModels.require(REF).toString();
        var first = new CliFixtures.Capture("");
        assertEquals(
                0,
                Main.run(
                        new String[] {
                            "instruct",
                            "-m",
                            model,
                            "Say hi",
                            "--temp",
                            "0",
                            "-n",
                            "16",
                            "--cache",
                            cache.toString()
                        },
                        first.io,
                        ModelStore.of(dir)),
                first.err());
        long size = Files.size(cache);
        var second = new CliFixtures.Capture("");
        assertEquals(
                0,
                Main.run(
                        new String[] {
                            "instruct",
                            "-m",
                            model,
                            "Say hi",
                            "--temp",
                            "0",
                            "-n",
                            "16",
                            "--cache-ro",
                            cache.toString()
                        },
                        second.io,
                        ModelStore.of(dir)),
                second.err());
        assertTrue(second.err().contains("restored"));
        assertEquals(size, Files.size(cache));
    }

    private static ChatEngine engine(Options o, Arena weights) throws Exception {
        var files = o.resolve(ModelStore.standard());
        return new ChatEngine(
                AOT.load(files.model(), files.companions(), files.tokenizer(), weights),
                "test",
                o.cacheOptions());
    }
}
