package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The non-thinking generation prompt is a per-CHECKPOINT contract, not a family-wide one, and the
 * vocabulary cannot decide it: every Gemma 4 vocabulary carries the channel specials (the template
 * splits on them to strip thought out of history), but only some templates scaffold with them.
 *
 * <p>12B and 26B close an empty thought channel when thinking is off, so the model skips straight
 * to the answer. E2B's template ends its generation prompt at {@code <|turn>model\n} either way -
 * given the scaffold anyway, it answers in reasoning prose instead of skipping the thought.
 *
 * <p>Tokenizer-only (no weights): reads each GGUF's own chat_template and asserts the prompt the
 * port builds agrees with it.
 */
class Gemma4ThinkingScaffoldTest {

    /** E2B declares no non-thinking branch, so {@code thinking} must be a no-op for it. */
    @Test
    void e2bDoesNotScaffoldTheNonThinkingPrompt() throws IOException {
        Checkpoint e2b = Checkpoint.of(ModelFixture.GEMMA4_E2B_Q8.path());
        assertFalse(
                e2b.chatTemplate.contains("not enable_thinking"),
                "fixture changed: E2B's template now has a non-thinking branch");
        assertArrayEquals(
                e2b.prompt(true),
                e2b.prompt(false),
                "E2B's template ends the generation prompt at <|turn>model\\n in BOTH modes -"
                        + " scaffolding it anyway makes the model answer in reasoning prose");
    }

    /** 12B declares the branch, so thinking OFF must close an empty thought channel. */
    @Test
    void twelveBScaffoldsTheNonThinkingPromptWithAClosedThoughtChannel() throws IOException {
        Checkpoint m = Checkpoint.of(ModelFixture.GEMMA4_12B_Q8.path());
        assertTrue(
                m.chatTemplate.contains("not enable_thinking"),
                "fixture changed: 12B's template lost its non-thinking branch");

        int[] thinking = m.prompt(true);
        int[] noThink = m.prompt(false);
        assertTrue(noThink.length > thinking.length, "thinking=false must add the closed channel");
        assertArrayEquals(
                thinking,
                java.util.Arrays.copyOf(noThink, thinking.length),
                "the scaffold is a SUFFIX of <|turn>model\\n");

        int[] scaffold = java.util.Arrays.copyOfRange(noThink, thinking.length, noThink.length);
        assertEquals(
                SpecialTokens.require(m.tokenizer, "<|channel>"),
                scaffold[0],
                "scaffold must open the channel");
        assertEquals(
                SpecialTokens.require(m.tokenizer, "<channel|>"),
                scaffold[scaffold.length - 1],
                "scaffold must CLOSE the channel - an open one is a thought the model continues");
        assertEquals(
                "thought\n",
                m.tokenizer.decode(java.util.Arrays.copyOfRange(scaffold, 1, scaffold.length - 1)),
                "the channel is named 'thought' and is left empty");
    }

    /** A GGUF's tokenizer and chat_template, without its weights. */
    private record Checkpoint(
            Tokenizer tokenizer, String chatTemplate, Gemma4TurnTemplate template) {

        static Checkpoint of(Path gguf) throws IOException {
            assumeTrue(Files.exists(gguf), "fixture not downloaded: " + gguf);
            GGUF meta;
            try (FileChannel ch = FileChannel.open(gguf, StandardOpenOption.READ)) {
                meta = ModelLoader.readGguf(ch, gguf.toString());
            }
            Tokenizer tokenizer = Tokenizers.fromGGUF(meta);
            String source = meta.getString("tokenizer.chat_template");
            return new Checkpoint(
                    tokenizer, source, new Gemma4TurnTemplate(tokenizer, null, 0, source));
        }

        int[] prompt(boolean thinking) {
            List<Batch> batches = template.generationPrompt(thinking);
            assertEquals(1, batches.size(), "generation prompt is one batch");
            return ((Batch.Input.Tokens) batches.get(0).input()).ids();
        }
    }
}
