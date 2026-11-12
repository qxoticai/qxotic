package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The typed API's edges, over a canned model: what a caller can write, what each factory refuses,
 * how answers are keyed and compared, and what a degenerate distribution reads as. No weights.
 */
final class DecisionApiEdgeTest {

    enum One {
        ONLY
    }

    enum Team {
        BILLING,
        SHIPPING
    }

    enum Odd {
        _LEADING,
        TRAILING_,
        WITH_2_DIGITS,
        lower_case
    }

    /** Echoes a fixed row and records what it was asked. */
    static class Canned implements DecisionModel {
        float[] row = {0.5f, 0.5f};
        final List<Question> questions = new ArrayList<>();
        final List<List<Content>> inputs = new ArrayList<>();
        int capacity;

        @Override
        public ContextState newState(int contextCapacity) {
            capacity = contextCapacity;
            return new ContextState(16, 16, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                @Override
                protected void clearHistory() {}
            };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            // run inside the state, as every real model does: a closed state refuses here
            state.exclusively(() -> {});
            questions.add(question);
            inputs.add(input);
            return row;
        }
    }

    private final Canned model = new Canned();

    @Test
    void aQuestionFactoryRefusesWhatCannotBeAsked() {
        // blank text, in every kind
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(""));
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(" \t\n"));
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(null));
        assertThrows(IllegalArgumentException.class, () -> Question.choice("", "a", "b"));
        assertThrows(IllegalArgumentException.class, () -> Question.rating(null, "a", "b"));

        // fewer than two outcomes
        assertThrows(IllegalArgumentException.class, () -> Question.choice("q", "only"));
        assertThrows(IllegalArgumentException.class, () -> Question.choice("q", new String[0]));
        assertThrows(IllegalArgumentException.class, () -> Question.rating("q", "only"));
        assertThrows(IllegalArgumentException.class, () -> Question.rating("q", new String[0]));
        assertThrows(IllegalArgumentException.class, () -> Question.choice("q", One.class));
        assertThrows(IllegalArgumentException.class, () -> Question.rating("q", One.class));
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.choice("q", new LinkedHashMap<String, String>()));
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.rating("q", new LinkedHashMap<String, String>()));

        // a label the model cannot read
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.choice("q", List.of("a", "b"), key -> null));
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.choice("q", List.of("a", "b"), key -> "  "));
    }

    /** Enum constants are read lower case, underscores as spaces, whatever their spelling. */
    @Test
    void enumLabelsAreReadableWhateverTheirSpelling() {
        assertEquals(
                List.of(
                        new DecisionModel.Option("leading"),
                        new DecisionModel.Option("trailing"),
                        new DecisionModel.Option("with 2 digits"),
                        new DecisionModel.Option("lower case")),
                Question.rating("q", Odd.class).labels());
    }

    /** Asked twice in one call, a question holds the last answer. */
    @Test
    void oneInstanceAskedTwiceHoldsTheLastAnswer() {
        var refund = Question.predicate("Refund?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            model.row = new float[] {0.9f, 0.1f};
            Answers answers = decisions.askIndependently(state, "x", refund, refund);
            assertFalse(answers.get(refund).isYes(), "the last row is 0.1 yes");
            assertEquals(2, model.questions.size(), "asked twice");
        }
    }

    /** Answers are keyed by value: an equal question finds the answer, another one is refused. */
    @Test
    void anEqualQuestionFindsTheAnswer() {
        model.row = new float[] {0.25f, 0.75f};
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            Answers answers = decisions.askIndependently(state, "x", Question.predicate("Refund?"));
            assertEquals(0.75f, answers.get(Question.predicate("Refund?")).yes());
            IllegalArgumentException refused =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> answers.get(Question.predicate("Urgent?")));
            assertEquals("this question was not asked", refused.getMessage());
        }
    }

    /** An input is passed through as the caller's parts, in order. */
    @Test
    void anInputKeepsTheCallersParts() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            decisions.ask(state, "", q); // an empty text is a text
            assertEquals(List.of(new Content.Text("")), model.inputs.getLast());
            List<Content> parts = List.of(new Content.Text("a"), new Content.Text("b"));
            decisions.ask(state, parts, q);
            assertEquals(parts, model.inputs.getLast());
        }
    }

    /** A degenerate distribution still answers with the rule each accessor promises. */
    @Test
    void degenerateDistributionsAnswerByTheirRules() {
        // one outcome gets everything
        var certain = new Answer.Rating<>(List.of("a", "b", "c"), List.of(0f, 0f, 1f));
        assertEquals(2, certain.selectedIndex());
        assertEquals(1f, certain.confidence());
        assertEquals(2f, certain.mean(), 1e-6f);
        assertEquals(2, certain.medianIndex());

        // a uniform split: the earliest wins every tie
        var even = new Answer.Choice<>(List.of("a", "b", "c"), List.of(1 / 3f, 1 / 3f, 1 / 3f));
        assertEquals(0, even.selectedIndex());
        assertEquals(1 / 3f, even.confidence(), 1e-6f);
        assertEquals("a", even.selectedOption());

        // halves: the first level the running total reaches a half at; exactly a half is the
        // earlier level, the same rule selectedIndex uses for a tie
        var halves = new Answer.Rating<>(List.of("a", "b", "c", "d"), List.of(.1f, .4f, .3f, .2f));
        assertEquals(1, halves.medianIndex(), "0.1 + 0.4 reaches a half exactly");
        assertEquals(1.6f, halves.mean(), 1e-5f);
        var past = new Answer.Rating<>(List.of("a", "b", "c", "d"), List.of(.1f, .39f, .3f, .21f));
        assertEquals(2, past.medianIndex(), "0.49 < 0.5; + 0.3 >= 0.5");
        assertEquals(1.62f, past.mean(), 1e-5f);

        // an all-zero row (never produced by a model, but constructible) picks the first
        var zero = new Answer.Choice<>(List.of("a", "b"), List.of(0f, 0f));
        assertEquals(0, zero.selectedIndex());
        assertEquals(0f, zero.confidence());
    }

    @Test
    void anAnswerRefusesWhatDoesNotMatchItsQuestion() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new Answer.Choice<>(List.of("a", "b"), List.of(1f)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Answer.Rating<>(List.of("a"), List.of(0.5f, 0.5f)));
        // probabilities are copied: a caller's list cannot change an answer under it
        List<Float> mutable = new ArrayList<>(List.of(0.2f, 0.8f));
        var answer = new Answer.Choice<>(List.of("a", "b"), mutable);
        mutable.set(0, 0f);
        assertEquals(0.2f, answer.probabilities().get(0));
    }

    /** A typed answer is the question's own: a choice over an enum answers with its constants. */
    @Test
    void answersCarryTheirQuestionsType() {
        var which = Question.choice("Which?", Team.class);
        var howMuch = Question.rating("How much?", "low", "high");
        Decisions decisions = Decisions.of(model);
        try (ContextState state = model.newState(64)) {
            model.row = new float[] {0.1f, 0.9f};
            Answers both = decisions.askIndependently(state, "x", which, howMuch);
            Answer.Choice<Team> choice = both.get(which);
            assertEquals(Team.SHIPPING, choice.selectedOption());
            Answer.Rating<String> rating = both.get(howMuch);
            assertEquals("high", rating.selectedLevel());
            // a wrong-type read is the compiler's to catch; at runtime the erasure is the caller's
            assertEquals(2, choice.probabilities().size());
        }
    }

    /** A null model or state is refused where it is passed, with the argument named. */
    @Test
    void aNullModelIsRefusedNamingIt() {
        NullPointerException refused =
                assertThrows(NullPointerException.class, () -> Decisions.of(null));
        assertEquals("model", refused.getMessage());
        NullPointerException noState =
                assertThrows(
                        NullPointerException.class,
                        () -> Decisions.of(model).ask(null, "x", Question.predicate("?")));
        assertEquals("state", noState.getMessage());
    }

    /** One Decisions serves any number of states: closing one leaves the others usable. */
    @Test
    void closingOneStateLeavesAnotherUsable() {
        var q = Question.predicate("?");
        Decisions decisions = Decisions.of(model);
        ContextState first = model.newState(64);
        try (ContextState second = model.newState(64)) {
            decisions.ask(first, "x", q);
            decisions.ask(second, "x", q);
        }
        // the second closed; the first still answers
        assertEquals(0.5f, decisions.ask(first, "x", q).yes(), 1e-6f);
        first.close();
        assertThrows(IllegalStateException.class, () -> decisions.ask(first, "x", q));
    }

    /** A choice's options and its labels are separate: the caller acts on options. */
    @Test
    void choicesActOnKeysAndLabelWithDescriptions() {
        record Tool(String name) {}
        Tool search = new Tool("search"), mail = new Tool("mail");
        Question.Choice<Tool> question =
                Question.choice("Which tool?", List.of(search, mail), Tool::name);
        assertEquals(List.of(search, mail), question.options());
        assertEquals(
                List.of(new DecisionModel.Option("search"), new DecisionModel.Option("mail")),
                question.labels());

        // a described map keeps its order and its null descriptions
        var strings = new LinkedHashMap<String, String>();
        strings.put("low", null);
        strings.put("high", "Needs an answer today");
        assertEquals(
                List.of(
                        new DecisionModel.Option("low"),
                        new DecisionModel.Option("high", "Needs an answer today")),
                Question.rating("How urgent?", strings).labels());
    }
}
