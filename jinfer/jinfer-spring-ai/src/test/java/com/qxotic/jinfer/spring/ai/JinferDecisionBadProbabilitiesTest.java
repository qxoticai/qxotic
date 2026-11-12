package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

/**
 * A model that answers badly, through Spring AI: NaN, out-of-range values, rows of the wrong shape.
 * Spring's own answer builders validate probabilities, so the adapter must not repair a bad model
 * into a plausible answer; a caller sees either the model's numbers or a refusal.
 */
final class JinferDecisionBadProbabilitiesTest {

    /** Answers with whatever {@link #rows} says. */
    static class Hostile implements com.qxotic.jinfer.chat.DecisionModel {
        Function<Question, float[]> rows = q -> new float[q.options().size()];

        @Override
        public ContextState newState(int contextCapacity) {
            return new ContextState(8, 8, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                @Override
                protected void clearHistory() {}
            };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            return decideIndependently(state, input, List.of(question))[0];
        }

        @Override
        public float[][] decideIndependently(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            float[][] out = new float[questions.size()][];
            for (int i = 0; i < out.length; i++) out[i] = rows.apply(questions.get(i));
            return out;
        }
    }

    private static Map<String, DecisionQuestion> questions(Object... namesAndQuestions) {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        for (int i = 0; i < namesAndQuestions.length; i += 2)
            questions.put(
                    (String) namesAndQuestions[i], (DecisionQuestion) namesAndQuestions[i + 1]);
        return questions;
    }

    @Test
    void aNanProbabilityIsRefused() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {Float.NaN, Float.NaN};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            assertThrows(
                    RuntimeException.class,
                    () -> adapter.call("x", questions("q", YesNoQuestion.of("?"))));
        }
    }

    @Test
    void anOutOfRangeProbabilityIsRefused() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.5f, 1.5f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            assertThrows(
                    RuntimeException.class,
                    () -> adapter.call("x", questions("q", YesNoQuestion.of("?"))));
        }
        Hostile choice = new Hostile();
        choice.rows = q -> new float[] {0.75f, 0.75f, -0.5f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(choice).build()) {
            assertThrows(
                    RuntimeException.class,
                    () -> adapter.call("x", questions("c", ChoiceQuestion.of("?", "a", "b", "c"))));
        }
    }

    /** A row that does not sum to one passes through: it is the model's own numbers. */
    @Test
    void aRowThatDoesNotSumToOnePassesThrough() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.2f, 0.2f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            DecisionResponse response = adapter.call("x", questions("q", YesNoQuestion.of("?")));
            assertEquals(0.2, response.getYesNo("q").getProbability(), 1e-6);
            assertTrue(!response.getYesNo("q").isYes());
        }
    }

    /** A short row is not padded and a long one not truncated: the adapter fails loudly. */
    @Test
    void aRowOfTheWrongShapeIsNotRepaired() {
        for (float[] row : new float[][] {{0.5f}, {0.1f, 0.2f, 0.3f, 0.4f}}) {
            Hostile model = new Hostile();
            model.rows = q -> row;
            try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
                IllegalStateException refused =
                        assertThrows(
                                IllegalStateException.class,
                                () -> adapter.call("x", questions("q", YesNoQuestion.of("?"))));
                assertTrue(refused.getMessage().contains("for 2 options"), refused.getMessage());
            }
        }
    }

    /** A score's value stays on the scale, never past the top level. */
    @Test
    void aScoreValueStaysOnTheScale() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0f, 0f, 0f, 1f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            DecisionResponse response =
                    adapter.call("x", questions("s", ScoreQuestion.of("?", "a", "b", "c", "d")));
            assertEquals(3.0, response.getScore("s").getValue(), 1e-9);
            assertEquals(3, response.getScore("s").getNearestLevel());
        }
    }

    /** A choice's value is always one the caller offered, even on a flat row. */
    @Test
    void aChoiceValueIsAlwaysAnOfferedOption() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.25f, 0.25f, 0.25f, 0.25f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            DecisionResponse response =
                    adapter.call("x", questions("c", ChoiceQuestion.of("?", "a", "b", "c", "d")));
            // Spring AI's ChoiceAnswer carries the per-option probabilities, keyed by option
            assertEquals(
                    List.of("a", "b", "c", "d"),
                    List.copyOf(response.getChoice("c").getProbabilities().keySet()));
            assertTrue(
                    response.getChoice("c")
                            .getProbabilities()
                            .containsKey(response.getChoice("c").getValue()),
                    response.getChoice("c").getValue());
        }
    }
}
