package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.ContextState;
import java.util.List;

/**
 * A model that answers questions about an input with probabilities, nothing generated: yes or no,
 * one of several options, a level on a scale. The model family owns how a question is put to the
 * model - the prompt it was trained on, the tokens read for each option, the calibration - and may
 * put each kind in a form of its own.
 *
 * <p>The input is what is judged: text and media parts in order, {@link Content.Text} and {@link
 * Content.Media}; the other kinds of content belong to a conversation and are refused, and so is an
 * input of no part. A model reads the parts in their order as far as it was trained to, and says
 * where it does not. Media take the model's towers, a companion of its file ({@link
 * Models#loadDecisionModel(java.nio.file.Path, java.lang.foreign.Arena, java.util.Map)}).
 *
 * <p>An answer is one probability per {@linkplain Question#options option} of its question, in
 * option order, summing to 1. A call runs in a state from {@link #newState} and may overwrite
 * whatever the state holds; one call at a time per state, any number of states side by side over
 * the same model.
 */
public interface DecisionModel {

    /**
     * A state for calls whose input and questions together take up to {@code contextCapacity}
     * tokens; the caller closes it. As everywhere in jinfer, 0 is the default (4096 tokens, or the
     * model's maximum when that is smaller) and -1 the model's maximum. The state's memory grows
     * with the capacity, so size it to the inputs.
     *
     * @throws IllegalArgumentException for a capacity above the model's maximum, or below -1
     */
    ContextState newState(int contextCapacity);

    /**
     * Answers one question about {@code input}.
     *
     * @throws IllegalArgumentException when the input and the question do not fit {@code state},
     *     the question has more options than the model can tell apart, or the input holds a part
     *     that is neither text nor media
     * @throws UnsupportedOperationException when the model cannot take a media part of the input
     */
    float[] decide(ContextState state, List<Content> input, Question question);

    /**
     * Answers each question about {@code input} on its own: row {@code i} is what {@link #decide}
     * answers for question {@code i}, whatever else is asked. A model may run several of them in
     * one pass; by default they are asked one after another.
     */
    default float[][] decideEach(
            ContextState state, List<Content> input, List<? extends Question> questions) {
        float[][] answers = new float[questions.size()][];
        for (int i = 0; i < answers.length; i++)
            answers[i] = decide(state, input, questions.get(i));
        return answers;
    }

    /**
     * Answers the questions about {@code input} together, a row each. A model trained to may read
     * the input once for all of them, far faster than {@link #decideEach}; the questions are then
     * not kept apart, and an answer can shift with what is asked beside it and in which order. By
     * default the same as {@link #decideEach}.
     */
    default float[][] decideTogether(
            ContextState state, List<Content> input, List<? extends Question> questions) {
        return decideEach(state, input, questions);
    }

    /**
     * One question: {@code text} is what is asked, {@code options} the outcomes its answer weighs,
     * in the answer's order. The kinds differ in what the options mean; a model without a form of
     * its own for a kind answers it as a choice among its options.
     */
    sealed interface Question {
        String text();

        List<Option> options();
    }

    /**
     * Yes or no: the options {@code no} and {@code yes}, in that order, as false comes before true.
     * Each meaning says what counts as that side, or is null.
     */
    record Predicate(String text, String yesMeans, String noMeans) implements Question {
        public Predicate {
            requireText(text);
        }

        public Predicate(String text) {
            this(text, null, null);
        }

        @Override
        public List<Option> options() {
            return List.of(new Option("no", noMeans), new Option("yes", yesMeans));
        }
    }

    /** One of {@code options}, in the order the model reads them. */
    record Choice(String text, List<Option> options) implements Question {
        public Choice {
            requireText(text);
            options = List.copyOf(options);
            if (options.size() < 2)
                throw new IllegalArgumentException("a choice needs at least 2 options");
        }
    }

    /** A level of a scale: its {@code levels}, lowest first. */
    record Rating(String text, List<Option> levels) implements Question {
        public Rating {
            requireText(text);
            levels = List.copyOf(levels);
            if (levels.size() < 2)
                throw new IllegalArgumentException("a rating needs at least 2 levels");
        }

        /** The levels. */
        @Override
        public List<Option> options() {
            return levels;
        }
    }

    /** One option of a choice or one level of a rating: its label, and what it means, or null. */
    record Option(String label, String description) {
        public Option {
            if (label == null || label.isBlank()) throw new IllegalArgumentException("blank label");
        }

        public Option(String label) {
            this(label, null);
        }

        /** Options by label alone. */
        public static List<Option> of(String... labels) {
            return java.util.Arrays.stream(labels).map(Option::new).toList();
        }
    }

    private static void requireText(String text) {
        if (text == null || text.isBlank()) throw new IllegalArgumentException("blank text");
    }
}
