package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.TestLogits.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.memory.MemoryView;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * {@link Sampling} exists to stop four interchangeable numbers travelling as four arguments. These
 * guards are the reason: every one of them fires on a plausible mistake, not an absurd one.
 */
final class SamplingTest {

    @Test
    void topPIsAHalfOpenUnitInterval() {
        // the server validator states this same range to clients; keep the two in step
        new Sampling(1f, Float.MIN_VALUE, 0, 0f, null);
        new Sampling(1f, 1f, 0, 0f, null);
        for (float bad : new float[] {0f, -0.1f, 1.0001f, Float.NaN}) {
            Assertions.assertThrows(
                    IllegalArgumentException.class,
                    () -> new Sampling(1f, bad, 0, 0f, null),
                    "topP " + bad);
        }
    }

    @Test
    void aTransposedTemperatureAndTopPIsRejected() {
        // topP is a probability mass; temperature is unbounded. Swapping a typical pair
        // (0.7, 0.95) is silently plausible, which is exactly why the range check exists
        new Sampling(0.7f, 0.95f, 40, 0.05f, null); // the right way round
        assertThrows(
                IllegalArgumentException.class, () -> new Sampling(0.95f, 1.7f, 40, 0.05f, null));
        assertThrows(
                IllegalArgumentException.class, () -> new Sampling(-0.1f, 0.95f, 40, 0.05f, null));
    }

    /** 0 is not "no nucleus filter". A caller who means "off" says 1. */
    @Test
    void aZeroTopPIsRejectedRatherThanReadAsDisabled() {
        new Sampling(0.7f, 1f, 40, 0.05f, null);
        assertThrows(IllegalArgumentException.class, () -> new Sampling(0.7f, 0f, 40, 0.05f, null));
    }

    @Test
    void nanNeverPasses() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new Sampling(Float.NaN, 0.95f, 40, 0.05f, null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Sampling(0.7f, Float.NaN, 40, 0.05f, null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Sampling(0.7f, 0.95f, 40, Float.NaN, null));
    }

    @Test
    void negativeTopKAndOutOfRangeMinPAreRejected() {
        new Sampling(0.7f, 0.95f, 0, 0f, null); // 0 disables both, legitimately
        assertThrows(
                IllegalArgumentException.class, () -> new Sampling(0.7f, 0.95f, -1, 0.05f, null));
        assertThrows(
                IllegalArgumentException.class, () -> new Sampling(0.7f, 0.95f, 40, 1.5f, null));
    }

    /** A null argument keeps this record's value, so "the request said nothing" changes nothing. */
    @Test
    void overrideAppliesOnlyWhatIsGiven() {
        Sampling base = new Sampling(0.7f, 0.95f, 40, 0.05f, 42L);
        assertEquals(base, base.override(null, null, null, null, null));
        assertEquals(
                new Sampling(0.2f, 0.95f, 40, 0.05f, 7L),
                base.override(0.2f, null, null, null, 7L));
    }

    /**
     * A null seed means fresh randomness PER CALL, not one seed resolved once: a server that reused
     * a root would replay identical completions for identical prompts.
     */
    @Test
    void aNullSeedDrawsFreshRandomnessEveryTime() {
        Sampling sampling = new Sampling(1f, 1f, 0, 0f, null);
        assertNull(sampling.seed());
        // 4096 draws from a uniform-ish distribution over 64 tokens: two independent samplers
        // agreeing on all of them is not a thing that happens
        assertNotEquals(draws(sampling), draws(sampling));
        Sampling seeded = new Sampling(1f, 1f, 0, 0f, 42L);
        assertEquals(draws(seeded), draws(seeded), "a seeded stack must be reproducible");
    }

    @Test
    void temperatureZeroIsGreedyOutright() {
        assertSame(Sampler.ARGMAX, new Sampling(0f, 0.95f, 40, 0.05f, null).sampler(64));
    }

    private static List<Integer> draws(Sampling sampling) {
        Sampler sampler = sampling.sampler(64);
        MemoryView<?> logits = view(64);
        List<Integer> tokens = new ArrayList<>();
        for (int i = 0; i < 4096; i++) {
            tokens.add(sampler.sampleToken(logits));
        }
        return tokens;
    }
}
