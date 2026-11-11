package com.qxotic.jinfer.decision;

import java.util.List;

/**
 * A question's answer: the model's probabilities over the question's outcomes, in the question's
 * order, and the outcome they favour. Everything an answer offers is read off those probabilities;
 * how well calibrated they are is the model's.
 */
public sealed interface Answer {

    /** One probability per outcome, in the question's order, summing to 1. */
    List<Float> probabilities();

    /** The position of the most probable outcome; the earlier one of a tie. */
    default int selectedIndex() {
        List<Float> p = probabilities();
        int best = 0;
        for (int i = 1; i < p.size(); i++) if (p.get(i) > p.get(best)) best = i;
        return best;
    }

    /** The probability of the selected outcome: 1 is certain, an even split is 1/n. */
    default float confidence() {
        return probabilities().get(selectedIndex());
    }

    /** A yes/no answer: the probability of yes. Its outcomes are no, then yes. */
    record Predicate(float yes) implements Answer {

        /** The probability of no. */
        public float no() {
            return 1 - yes;
        }

        /** Whether yes is the more likely side. */
        public boolean isYes() {
            return yes > 0.5f;
        }

        @Override
        public List<Float> probabilities() {
            return List.of(no(), yes);
        }
    }

    /**
     * A choice's answer: the probability of each of the question's options, by position.
     *
     * @param <E> the option type
     * @param options the question's options, as asked
     */
    record Choice<E>(List<E> options, List<Float> probabilities) implements Answer {
        public Choice {
            options = List.copyOf(options);
            probabilities = List.copyOf(probabilities);
            requireSameSize(options, probabilities);
        }

        /** The most probable option: the one at {@link #selectedIndex}. */
        public E selectedOption() {
            return options.get(selectedIndex());
        }
    }

    /**
     * A rating's answer: the probability of each level of the scale, lowest level first.
     *
     * @param <E> the level type
     * @param levels the question's levels, as asked
     */
    record Rating<E>(List<E> levels, List<Float> probabilities) implements Answer {
        public Rating {
            levels = List.copyOf(levels);
            probabilities = List.copyOf(probabilities);
            requireSameSize(levels, probabilities);
        }

        /** The most probable level: the one at {@link #selectedIndex}. */
        public E selectedLevel() {
            return levels.get(selectedIndex());
        }

        /** The position on the scale as a number: 1.4 is mostly level 1, some level 2. */
        public float mean() {
            float mean = 0;
            for (int i = 0; i < probabilities.size(); i++) mean += i * probabilities.get(i);
            return mean;
        }

        /**
         * The position of the first level at which the cumulative probability reaches a half: the
         * middle of the distribution, where most of the probability mass has accumulated.
         *
         * <p>{@link #mean()} is the level every face reports (LangChain4j's {@code mean()}, Spring
         * AI's {@code getValue()}, the server's {@code score}); use this for a level the model gave
         * mass to, which a mean can miss when the distribution is split between distant levels.
         */
        public int medianIndex() {
            float mass = 0;
            for (int i = 0; i < probabilities.size(); i++)
                if ((mass += probabilities.get(i)) >= 0.5f) return i;
            return probabilities.size() - 1;
        }
    }

    private static void requireSameSize(List<?> outcomes, List<Float> probabilities) {
        if (outcomes.size() != probabilities.size())
            throw new IllegalArgumentException(
                    outcomes.size() + " outcomes and " + probabilities.size() + " probabilities");
    }
}
