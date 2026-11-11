package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

final class DecisionsTest {

    enum Team {
        BILLING,
        SHIPPING,
        TECH_SUPPORT
    }

    enum Severity {
        LOW,
        HIGH,
        CRITICAL
    }

    /** Answers with canned probabilities and keeps what it was asked, about what, and how. */
    static final class Canned implements DecisionModel {
        float[] row;
        final List<String> asked = new ArrayList<>();
        final List<DecisionModel.Question> questions = new ArrayList<>();
        List<Content> input;
        boolean together;
        ContextState state;
        int requested;

        @Override
        public ContextState newState(int contextCapacity) {
            requested = contextCapacity;
            return state =
                    new ContextState(64, 64, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                        @Override
                        protected void clearHistory() {}
                    };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            this.input = input;
            asked.add(question.toString());
            questions.add(question);
            return row;
        }

        @Override
        public float[][] decideTogether(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            together = true;
            return decideEach(state, input, questions);
        }
    }

    private final Canned model = new Canned();
    private final Decisions decisions = Decisions.over(model, 64);

    @Test
    void aPredicateAnswersYesOrNo() {
        model.row = new float[] {0.1f, 0.9f};
        Answer.Predicate a =
                decisions.ask("x", Question.predicate("Refund?", "Asks for money back", null));

        assertEquals(0.9f, a.yes());
        assertEquals(0.1f, a.no(), 1e-6f);
        assertTrue(a.isYes());
        assertEquals(List.of(a.no(), 0.9f), a.probabilities()); // no, yes
        assertEquals(1, a.selectedIndex());
        assertEquals(0.9f, a.confidence());
        assertEquals(
                List.of(
                        new DecisionModel.Predicate("Refund?", "Asks for money back", null)
                                .toString()),
                model.asked);

        model.row = new float[] {0.75f, 0.25f};
        Answer no = decisions.ask("x", Question.predicate("Refund?"));
        assertFalse(((Answer.Predicate) no).isYes());
        assertEquals(0, no.selectedIndex());
        assertEquals(0.75f, no.confidence());
    }

    @Test
    void exactlyAHalfIsNotYes() {
        // the rule is "more likely than not": a half is not
        assertFalse(new Answer.Predicate(0.5f).isYes());
        assertTrue(new Answer.Predicate(Math.nextUp(0.5f)).isYes());
        assertTrue(new Answer.Predicate(0.5f).no() == 0.5f);
    }

    @Test
    void aChoiceOverAnEnumAnswersWithItsConstant() {
        model.row = new float[] {0.2f, 0.7f, 0.1f};
        Answer.Choice<Team> a = decisions.ask("x", Question.choice("Which team?", Team.class));

        assertEquals(Team.SHIPPING, a.selectedOption());
        assertEquals(1, a.selectedIndex());
        assertEquals(0.7f, a.confidence());
        assertEquals(List.of(Team.BILLING, Team.SHIPPING, Team.TECH_SUPPORT), a.options());
        assertEquals(List.of(0.2f, 0.7f, 0.1f), a.probabilities());
        // the model reads the constants in lower case with spaces
        assertEquals(
                List.of(
                        new DecisionModel.Choice(
                                        "Which team?",
                                        Option.of("billing", "shipping", "tech support"))
                                .toString()),
                model.asked);
    }

    @Test
    void describedOptionsKeepTheirOrderAndDescriptions() {
        LinkedHashMap<String, String> options = new LinkedHashMap<>();
        options.put("billing", "Charges and refunds");
        options.put("other", null);
        model.row = new float[] {0.25f, 0.75f};

        assertEquals(
                "other",
                decisions.ask("x", Question.choice("Which team?", options)).selectedOption());
        assertEquals(
                List.of(
                        new DecisionModel.Choice(
                                        "Which team?",
                                        List.of(
                                                new Option("billing", "Charges and refunds"),
                                                new Option("other")))
                                .toString()),
                model.asked);
    }

    @Test
    void aRatingOverAnEnumAnswersWithItsConstant() {
        model.row = new float[] {0.1f, 0.3f, 0.6f};
        Answer.Rating<Severity> a =
                decisions.ask("x", Question.rating("How severe?", Severity.class));

        assertEquals(Severity.CRITICAL, a.selectedLevel());
        assertEquals(2, a.selectedIndex());
        assertEquals(0.6f, a.confidence());
        assertEquals(1.5f, a.mean(), 1e-6f);
        assertEquals(2, a.medianIndex());
        assertEquals(
                List.of(
                        new DecisionModel.Rating(
                                        "How severe?", Option.of("low", "high", "critical"))
                                .toString()),
                model.asked);
    }

    @Test
    void severalQuestionsAreAnsweredTogetherAndFoundByTheirQuestion() {
        var refund = Question.predicate("Refund?");
        var urgency = Question.rating("How urgent?", "Can wait", "Today", "Now");
        model.row = new float[] {0.1f, 0.3f, 0.6f}; // a predicate reads its first two as no, yes

        Answers answers = decisions.askEach("x", refund, urgency);

        assertFalse(answers.get(refund).isYes());
        assertEquals("Now", answers.get(urgency).selectedLevel());
        assertEquals(2, model.asked.size());
        assertThrows(
                IllegalArgumentException.class, () -> answers.get(Question.predicate("Refund?")));
    }

    @Test
    void anInputIsATextOrItsPartsAndTogetherIsAskedFor() {
        model.row = new float[] {0.5f, 0.5f};
        var q = Question.predicate("?");
        decisions.ask("ticket", q);
        assertEquals(List.of(new Content.Text("ticket")), model.input);

        List<Content> parts = List.of(new Content.Text("a"), new Content.Text("b"));
        decisions.askEach(parts, q, q);
        assertEquals(parts, model.input);
        assertFalse(model.together);
        decisions.askTogether(parts, q, q);
        assertTrue(model.together);
    }

    @Test
    void keysAreWhatTheCallerActsOnAndLabelsWhatTheModelReads() {
        record Tool(String name, String doc) {}
        List<Tool> tools =
                List.of(new Tool("search", "Finds documents"), new Tool("mail", "Sends"));
        model.row = new float[] {0.3f, 0.7f};
        Answer.Choice<Tool> picked =
                decisions.ask("x", Question.choice("Which tool?", tools, Tool::doc));
        assertEquals(tools.get(1), picked.selectedOption());
        assertEquals(
                new DecisionModel.Choice("Which tool?", Option.of("Finds documents", "Sends")),
                model.questions.getLast());

        var scale = new LinkedHashMap<String, String>();
        scale.put("low", "Can wait");
        scale.put("high", null);
        model.row = new float[] {0.6f, 0.4f};
        assertEquals(
                "low", decisions.ask("x", Question.rating("How urgent?", scale)).selectedLevel());
        assertEquals(
                new DecisionModel.Rating(
                        "How urgent?", List.of(new Option("low", "Can wait"), new Option("high"))),
                model.questions.getLast());
    }

    @Test
    void tiesAndHalvesResolveToTheEarlierLevel() {
        Answer.Rating<String> even =
                new Answer.Rating<>(List.of("a", "b", "c"), List.of(0.5f, 0.5f, 0f));
        assertEquals(0, even.selectedIndex(), "a tie picks the earlier");
        assertEquals(0, even.medianIndex(), "the first level the running total reaches a half at");
        assertEquals(0.5f, even.mean(), 1e-6f);
        assertThrows(
                IllegalArgumentException.class,
                () -> new Answer.Choice<>(List.of("a", "b"), List.of(1f)),
                "an answer has a probability per option");
    }

    @Test
    void theCapacityIsTheModelsRuleAndClosingReleasesTheState() {
        Decisions closing = Decisions.over(model);
        assertEquals(0, model.requested, "0: the model's default");
        assertTrue(model.state.isAlive());
        closing.close();
        assertFalse(model.state.isAlive());
        Decisions.over(model, 2048).close();
        assertEquals(2048, model.requested);
    }

    /** A question that cannot be asked fails where it is written, not where it is asked. */
    @Test
    void aMalformedQuestionFailsAtConstruction() {
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(" "));
        assertThrows(IllegalArgumentException.class, () -> Question.choice(null, "a", "b"));
        assertThrows(IllegalArgumentException.class, () -> Question.choice("Which?", "only"));
        assertThrows(IllegalArgumentException.class, () -> Question.rating("How much?", "one"));
        assertEquals(0, model.asked.size(), "nothing reached the model");
    }
}
