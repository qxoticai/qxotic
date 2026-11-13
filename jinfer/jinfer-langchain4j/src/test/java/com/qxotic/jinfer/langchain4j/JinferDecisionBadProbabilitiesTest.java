package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jota.memory.MemoryAllocators;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * A model that answers badly: NaN, values outside 0..1, a row of the wrong length, a row that does
 * not sum to one. The framework validates probabilities, so the adapter must not launder garbage
 * into a plausible-looking answer - each case either passes through unchanged (a model's own
 * numbers) or is refused by the framework, never silently repaired.
 */
final class JinferDecisionBadProbabilitiesTest {

    /** Answers with whatever {@link #rows} says, as it says it. */
    static class Hostile implements DecisionModel {
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

    private static DecisionRequest oneYesNo() {
        return DecisionRequest.builder().input("x").question("q", YesNoQuestion.from("?")).build();
    }

    /** A NaN probability is not a number a caller can act on: the framework refuses it. */
    @Test
    void aNanProbabilityIsRefused() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {Float.NaN, Float.NaN};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            // langchain4j validates every probability it is handed: NaN never reaches a caller
            assertThrows(RuntimeException.class, () -> adapter.decide(oneYesNo()));
        }
    }

    /**
     * A probability above 1 is refused on every kind. A negative one is refused too, wherever the
     * value surfaces: a yes/no answer carries only P(yes), so a negative P(no) never reaches a
     * caller and passing it through is the framework's own shape, not a repair.
     */
    @Test
    void anOutOfRangeProbabilityIsRefused() {
        for (float[] row : new float[][] {{0.5f, 1.5f}, {2f, -1f}}) {
            Hostile model = new Hostile();
            model.rows = q -> row;
            try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
                assertThrows(
                        RuntimeException.class,
                        () -> adapter.decide(oneYesNo()),
                        List.of(row[0], row[1]).toString());
            }
        }
        // a negative probability on a choice DOES surface: refused, not clamped to zero
        Hostile choice = new Hostile();
        choice.rows = q -> new float[] {-0.5f, 0.75f, 0.75f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(choice).build()) {
            assertThrows(
                    RuntimeException.class,
                    () ->
                            adapter.decide(
                                    DecisionRequest.builder()
                                            .input("x")
                                            .question(
                                                    "c",
                                                    ChoiceQuestion.from(
                                                            "?", List.of("a", "b", "c")))
                                            .build()));
        }
    }

    /** A row that does not sum to one passes through: it is the model's own numbers. */
    @Test
    void aRowThatDoesNotSumToOnePassesThrough() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.2f, 0.2f}; // sums to 0.4
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            assertEquals(0.2, adapter.decide(oneYesNo()).yesNo("q").probability(), 1e-6);
        }
    }

    /** A short row is the model's bug: the adapter must fail loudly, not invent an option. */
    @Test
    void aShortRowIsNotPadded() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.5f}; // one value for a two-option question
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            assertThrows(RuntimeException.class, () -> adapter.decide(oneYesNo()));
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
                        assertThrows(IllegalStateException.class, () -> adapter.decide(oneYesNo()));
                assertTrue(refused.getMessage().contains("for 2 options"), refused.getMessage());
            }
        }
    }

    /**
     * A scale answer is a mean in [0, levels-1]: an out-of-range level is refused, not reported.
     */
    @Test
    void aScaleMeanStaysOnTheScale() {
        Hostile model = new Hostile();
        // the last level carries all the mass: the mean is exactly the top level, never past it
        model.rows = q -> new float[] {0f, 0f, 0f, 1f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            var response =
                    adapter.decide(
                            DecisionRequest.builder()
                                    .input("x")
                                    .question(
                                            "s",
                                            ScaleQuestion.from("?", List.of("a", "b", "c", "d")))
                                    .build());
            assertEquals(3.0, response.scale("s").mean(), 1e-9);
        }
    }

    /** A choice's value is always one the caller offered, even on a tie or a flat row. */
    @Test
    void aChoiceValueIsAlwaysAnOfferedOption() {
        Hostile model = new Hostile();
        model.rows = q -> new float[] {0.25f, 0.25f, 0.25f, 0.25f};
        try (JinferDecisionModel adapter = JinferDecisionModel.builder().model(model).build()) {
            var response =
                    adapter.decide(
                            DecisionRequest.builder()
                                    .input("x")
                                    .question(
                                            "c",
                                            ChoiceQuestion.from("?", List.of("a", "b", "c", "d")))
                                    .build());
            assertTrue(response.choice("c").options().contains(response.choice("c").value()));
            assertEquals("a", response.choice("c").value(), "the first of a tie");
            // a flat row has no margin: nothing leads, which is the escalation signal itself
            assertEquals(0.0, response.choice("c").margin(), 1e-9);
        }
    }
}
