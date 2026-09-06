package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.toknroll.IntSequence;
import org.junit.jupiter.api.Test;

class InstructTest {

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
}
