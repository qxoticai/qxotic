package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.chat.LoadedModel;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Public syntax and boundary values: failures must name the user's option, not a later model load.
 */
class ArgumentMatrixTest {
    @ParameterizedTest(name = "{0} {1} {2}")
    @CsvSource({
        "instruct,--threads,0",
        "instruct,--threads,-1",
        "instruct,--threads,2147483648",
        "instruct,--batch-capacity,0",
        "instruct,--context-capacity,-1",
        "instruct,--temp,-0.1",
        "instruct,--temp,Infinity",
        "instruct,--temp,not-a-number",
        "instruct,--top-p,0",
        "instruct,--top-p,1.01",
        "instruct,--top-p,NaN",
        "instruct,--top-k,-1",
        "instruct,--min-p,-0.01",
        "instruct,--min-p,1.01",
        "instruct,--max-output-tokens,-2",
        "instruct,--max-reasoning-tokens,-2",
        "instruct,--speculation-depth,-1",
        "instruct,--speculation-depth,9",
        "instruct,--seed,9223372036854775808",
        "instruct,--think,sometimes",
        "instruct,--color,rainbow",
        "server,--port,-1",
        "server,--port,65536",
        "server,--concurrency,0",
        "server,--concurrency,2147483647",
        "server,--queue-depth,-1",
        "server,--max-body-mb,0",
        "server,--max-body-mb,-1",
        "server,--write-timeout,0",
        "server,--write-timeout,-1",
        "server,--request-timeout,-1",
        "server,--request-timeout,9223372036854775808"
    })
    void invalidNumbersAndEnumsNameTheirOption(String command, String option, String value) {
        var args = new ArrayList<>(List.of(command, "-m", "unused", option, value));
        if (command.equals("instruct")) args.add("hello");
        var failure =
                assertThrows(Options.Usage.class, () -> Options.parse(args.toArray(String[]::new)));
        assertTrue(failure.getMessage().contains(option), failure.getMessage());
        assertTrue(
                failure.getMessage().contains(value),
                "rejected value is missing: " + failure.getMessage());
    }

    @Test
    void completeGenerationSettingsOverrideModelRecommendations() {
        Options o =
                Options.parse(
                        "--model",
                        "unused",
                        "--threads",
                        "1",
                        "--batch-capacity",
                        "1",
                        "--context-capacity",
                        "0",
                        "--temp",
                        "0.25",
                        "instruct",
                        "hello",
                        "--top-p",
                        "0.8",
                        "--top-k",
                        "12",
                        "--min-p",
                        "0.1",
                        "--seed",
                        "-9223372036854775808",
                        "--max-output-tokens",
                        "0",
                        "--max-reasoning-tokens",
                        "-1",
                        "--speculation-depth",
                        "8");
        var sampling = o.sampling(new LoadedModel.SamplingDefaults(0.9f, 0.9f, 40, 0.05f));
        assertEquals(0.25f, sampling.temperature());
        assertEquals(0.8f, sampling.topP());
        assertEquals(12, sampling.topK());
        assertEquals(0.1f, sampling.minP());
        assertEquals(Long.MIN_VALUE, o.seed);
        assertEquals(0, o.contextCapacity);
        assertEquals(1, o.batchCapacity);
        assertEquals(0, o.maxOutputTokens);
        assertEquals(-1, o.maxReasoningTokens);
        assertEquals(8, o.speculationDepth);
    }

    @Test
    void disablingSamplingAndOverridingCacheModesAreExplicit() {
        Options o =
                Options.parse(
                        "instruct",
                        "-m",
                        "unused",
                        "hello",
                        "--top-p",
                        "1",
                        "--top-k",
                        "0",
                        "--min-p",
                        "0",
                        "-n",
                        "-1",
                        "--speculation-depth",
                        "0",
                        "--cache",
                        "first.jkv",
                        "--cache-ro",
                        "second.jkv");
        assertEquals(0, o.topK);
        assertEquals(0f, o.minP);
        assertEquals(-1, o.maxOutputTokens);
        assertEquals(0, o.speculationDepth);
        assertTrue(o.promptCacheReadOnly);
        assertEquals("second.jkv", o.promptCache.toString());
        assertFalse(
                Options.parse(
                                "instruct",
                                "-m",
                                "m",
                                "hi",
                                "--cache-ro",
                                "first",
                                "--cache",
                                "second")
                        .promptCacheReadOnly);
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "--threads",
                "--batch-capacity",
                "--temp",
                "--seed",
                "--with",
                "--model",
                "--context-capacity"
            })
    void missingSharedValuesAreReportedBeforeAnyModelWork(String flag) {
        var failure =
                assertThrows(
                        Options.Usage.class, () -> Options.parse("chat", "-m", "unused", flag));
        assertTrue(
                failure.getMessage().contains("Missing argument for option " + flag),
                failure.getMessage());
    }

    @ParameterizedTest
    @CsvSource({
        "speak,--output",
        "speak,--speed",
        "server,--host",
        "server,--port",
        "server,--api-key",
        "server,--queue-depth",
        "server,--request-timeout",
        "transcribe,--theme",
        "instruct,--prompt"
    })
    void missingApplicationValuesNameTheFlag(String command, String flag) {
        var failure =
                assertThrows(
                        Options.Usage.class, () -> Options.parse(command, "-m", "unused", flag));
        assertTrue(
                failure.getMessage().contains("Missing argument for option " + flag),
                failure.getMessage());
    }

    @Test
    void switchesNeverAcceptAccidentalAssignments() {
        for (String[] args :
                new String[][] {
                    {"speak", "-m", "m", "hello", "--play=false"},
                    {"server", "-m", "m", "--no-grammar=false"},
                    {"transcribe", "-m", "m", "-", "--raw-pcm=false"},
                    {"instruct", "-m", "m", "hi", "--no-stream=true"},
                    {"instruct", "-m", "m", "hi", "--stream=off"},
                    {"pull", "--force=false", "owner/repo:Q8_0"}
                })
            assertThrows(Options.Usage.class, () -> Options.parse(args), String.join(" ", args));
    }

    @Test
    void explicitBooleanSwitchesAndLegacySpellingsHaveClearPrecedence() {
        Options modern =
                Options.parse(
                        "instruct",
                        "-m",
                        "m",
                        "hi",
                        "--stream",
                        "--no-stream",
                        "--echo",
                        "--no-echo");
        assertFalse(modern.stream);
        assertFalse(modern.echo);
        for (String value : List.of("false", "OFF")) {
            Options old =
                    Options.parse(
                            "-m", "m", "--prompt", "hi", "--stream", value, "--echo=" + value);
            assertFalse(old.stream);
            assertFalse(old.echo);
        }
        assertTrue(Options.parse("-m", "m", "--prompt", "hi", "--echo", "ON").echo);
        assertThrows(
                Options.Usage.class,
                () -> Options.parse("-m", "m", "--prompt", "hi", "--echo=maybe"));
    }

    @Test
    void legacyModesRemainTokenAwareAndCannotBeMixedWithVerbs() {
        for (String flag : List.of("--chat", "--interactive", "-i"))
            assertEquals("chat", Options.parse("-m", "m", flag).command);
        assertEquals(
                "instruct", Options.parse("-m", "m", "--instruct", "--prompt", "hello").command);
        for (String[] args :
                new String[][] {
                    {"chat", "-m", "m", "--chat"},
                    {"speak", "-m", "m", "--speak", "hi"},
                    {"-m", "m", "--chat", "--server"},
                    {"-m", "m", "--transcribe", "-", "--speak", "hi"}
                }) assertThrows(Options.Usage.class, () -> Options.parse(args));
        assertEquals("server", Options.parse("instruct", "-m", "m", "server").input);
        assertEquals("--help", Options.parse("speak", "-m", "m", "--", "--help").input);
    }

    @Test
    void unsupportedAndDuplicateInputsNeverGetSilentlyIgnored() {
        for (String[] args :
                new String[][] {
                    {"chat", "-m", "m", "hello"},
                    {"server", "-m", "m", "hello"},
                    {"instruct", "-m", "m", "--prompt", "first", "second"},
                    {"speak", "-m", "m", "first", "second"},
                    {"cache-info", "a", "b"},
                    {"transcribe", "-m", "m", "audio.wav", "--theme", "mint"},
                    {"transcribe", "-m", "m", "-", "--raw-pcm", "--theme", "missing"},
                    {"speak", "-m", "m", "hi", "--with", "tokenizer=other.gguf"},
                    {"list", "--threads", "2"},
                    {"pull", "owner/repo:Q8_0", "--temp", "0"},
                    {"chat", "--model="},
                    {"server", "-m", "m", "--host="}
                })
            assertThrows(Options.Usage.class, () -> Options.parse(args), String.join(" ", args));
    }
}
