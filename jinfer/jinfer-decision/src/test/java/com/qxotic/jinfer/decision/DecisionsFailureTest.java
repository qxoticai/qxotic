package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * The typed API under misuse: a state that refuses, a model that throws, media and conversation
 * parts in a decision's input, and a null question list. No weights: every case here is the
 * wrapper's own.
 */
final class DecisionsFailureTest {

    /** Fails on demand, and records how many calls reached it. */
    static class Flaky implements DecisionModel {
        final AtomicInteger calls = new AtomicInteger();
        volatile RuntimeException failure;
        volatile float[] row = {0.25f, 0.75f};

        @Override
        public ContextState newState(int contextCapacity) {
            return new ContextState(16, 16, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                @Override
                protected void clearHistory() {}
            };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            state.exclusively(() -> {}); // a closed state refuses here, as a real model's does
            calls.incrementAndGet();
            if (failure != null) throw failure;
            return row;
        }
    }

    private final Flaky model = new Flaky();

    /** A model that throws leaves the wrapper usable: the failure is the caller's, not a poison. */
    @Test
    void aModelsFailureLeavesTheDecisionsUsable() {
        var q = Question.predicate("Refund?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            model.failure = new IllegalStateException("inference blew up");
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> decisions.ask(state, "x", q));
            assertEquals("inference blew up", thrown.getMessage());

            model.failure = null;
            assertTrue(decisions.ask(state, "x", q).isYes());
            assertEquals(2, model.calls.get(), "the failing call and the good one both reached it");
        }
    }

    /**
     * An IllegalArgumentException from the model (too long, too many options) reaches the caller.
     */
    @Test
    void aModelsRefusalReachesTheCallerUnchanged() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            model.failure =
                    new IllegalArgumentException("the input and its questions take 9000 tokens");
            IllegalArgumentException thrown =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> decisions.askIndependently(state, "x", q, q));
            assertTrue(thrown.getMessage().contains("9000 tokens"), thrown.getMessage());
        }
    }

    /** A closed state refuses every entry point, and the message is the state's own. */
    @Test
    void everyEntryPointRefusesAfterClose() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        ContextState state = model.newState(64);
        state.close();
        assertThrows(IllegalStateException.class, () -> decisions.ask(state, "x", q));
        assertThrows(IllegalStateException.class, () -> decisions.askIndependently(state, "x", q));
        assertThrows(IllegalStateException.class, () -> decisions.ask(state, "x", q, q));
        assertThrows(
                IllegalStateException.class,
                () -> decisions.ask(state, List.of(new Content.Text("x")), q));
        // idempotent: a second close is a no-op, not a second failure
        state.close();
    }

    /** Content parts a decision model cannot judge are refused by the model, not dropped. */
    @Test
    void conversationPartsAreTheModelsToRefuse() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            // the wrapper hands every part through: the model decides what it reads
            decisions.ask(state, List.of(new Content.Reasoning("hm"), new Content.Text("x")), q);
            assertEquals(1, model.calls.get(), "the wrapper does not pre-filter parts");

            decisions.ask(
                    state,
                    List.of(new Content.ToolResult("call_1", "42"), new Content.Text("x")),
                    q);
            assertEquals(2, model.calls.get());
        }
    }

    /** An empty input list is the model's to refuse; null is the caller's mistake, named. */
    @Test
    void aNullInputIsRefusedNamingIt() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            assertThrows(
                    NullPointerException.class,
                    () -> decisions.ask(state, (List<Content>) null, q));
            assertThrows(NullPointerException.class, () -> decisions.ask(state, (String) null, q));
            assertEquals(0, model.calls.get(), "nothing reached the model");
        }
    }

    /** Null questions and null arrays are refused before any inference. */
    @Test
    void nullQuestionsAreRefused() {
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            assertThrows(
                    NullPointerException.class,
                    () -> decisions.ask(state, "x", (Question<?>) null));
            assertThrows(
                    NullPointerException.class,
                    () -> decisions.askIndependently(state, "x", (Question<?>[]) null));
            assertThrows(
                    NullPointerException.class,
                    () -> decisions.ask(state, "x", (Question<?>[]) null));
            assertEquals(0, model.calls.get(), "nothing reached the model");
        }
    }

    /** A question array holding null is refused where it is lowered, before any inference. */
    @Test
    void anArrayWithANullQuestionIsRefused() {
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            Question<?>[] withNull = {Question.predicate("A?"), null};
            assertThrows(
                    NullPointerException.class,
                    () -> decisions.askIndependently(state, "x", withNull));
        }
    }
}
