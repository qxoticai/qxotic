package com.qxotic.jinfer.decision;

import com.qxotic.jinfer.chat.DecisionModel.Option;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.SequencedMap;
import java.util.function.Function;

/**
 * One typed question about an input. The answer type follows the question type: a {@link Predicate}
 * answers with {@link Answer.Predicate}, a {@link Choice} with {@link Answer.Choice}, a {@link
 * Rating} with {@link Answer.Rating}. Ask with {@link Decisions#ask}.
 *
 * @param <A> the answer this question produces
 */
public sealed interface Question<A extends Answer> {

    /** What is asked. */
    String text();

    /** A yes/no question. */
    static Predicate predicate(String text) {
        return new Predicate(text, null, null);
    }

    /** A yes/no question, each side described: what counts as yes, what counts as no. */
    static Predicate predicate(String text, String yesMeans, String noMeans) {
        return new Predicate(text, yesMeans, noMeans);
    }

    /**
     * Pick one constant of {@code type}, in declaration order. The model reads each constant's name
     * in lower case with spaces: {@code TECH_SUPPORT} as "tech support".
     */
    static <E extends Enum<E>> Choice<E> choice(String text, Class<E> type) {
        return choice(text, List.of(type.getEnumConstants()), Question::readable);
    }

    /** Pick one of {@code options}. */
    static Choice<String> choice(String text, String... options) {
        return new Choice<>(text, List.of(options), Option.of(options));
    }

    /** Pick one option by label, each with its description. */
    static Choice<String> choice(String text, SequencedMap<String, String> descriptions) {
        return new Choice<>(text, List.copyOf(descriptions.keySet()), described(descriptions));
    }

    /**
     * Pick one of {@code options}, whatever the caller acts on, each shown by its {@code label}.
     */
    static <E> Choice<E> choice(String text, List<E> options, Function<? super E, String> label) {
        return new Choice<>(text, options, labelled(options, label));
    }

    /** Where on the scale of {@code levels}, lowest first, the input falls. */
    static Rating<String> rating(String text, String... levels) {
        return new Rating<>(text, List.of(levels), Option.of(levels));
    }

    /**
     * The constants of {@code type} as the scale, lowest first in declaration order, read by the
     * model as {@link #choice(String, Class)} reads them.
     */
    static <E extends Enum<E>> Rating<E> rating(String text, Class<E> type) {
        List<E> levels = List.of(type.getEnumConstants());
        return new Rating<>(text, levels, labelled(levels, Question::readable));
    }

    /** Levels by label, lowest first, each with its description. */
    static Rating<String> rating(String text, SequencedMap<String, String> descriptions) {
        return new Rating<>(text, List.copyOf(descriptions.keySet()), described(descriptions));
    }

    /** Yes or no; each description says what counts as that side, or is null. */
    record Predicate(String text, String yesMeans, String noMeans)
            implements Question<Answer.Predicate> {
        public Predicate {
            requireText(text);
        }
    }

    /**
     * Pick one of {@code options}. Order is part of the question: it is the order the model reads
     * them in.
     *
     * @param <E> what the caller acts on: an enum constant, a tool, a document
     * @param labels how the model reads each option, in the same order
     */
    record Choice<E>(String text, List<E> options, List<Option> labels)
            implements Question<Answer.Choice<E>> {
        public Choice {
            requireText(text);
            options = List.copyOf(options);
            labels = List.copyOf(labels);
            requireSameSize(options, labels);
            if (options.size() < 2)
                throw new IllegalArgumentException("a choice needs at least 2 options");
        }
    }

    /**
     * Where on the scale of {@code levels}, lowest first.
     *
     * @param <E> what the caller reads a level as: a string, an enum constant
     * @param labels how the model reads each level, in the same order
     */
    record Rating<E>(String text, List<E> levels, List<Option> labels)
            implements Question<Answer.Rating<E>> {
        public Rating {
            requireText(text);
            levels = List.copyOf(levels);
            labels = List.copyOf(labels);
            requireSameSize(levels, labels);
            if (levels.size() < 2)
                throw new IllegalArgumentException("a rating needs at least 2 levels");
        }
    }

    private static String readable(Enum<?> constant) {
        // a leading or trailing underscore would otherwise reach the model as a stray space
        return constant.name().toLowerCase(Locale.ROOT).replace('_', ' ').strip();
    }

    private static <E> List<Option> labelled(List<E> values, Function<? super E, String> label) {
        List<Option> labels = new ArrayList<>(values.size());
        for (E value : values) labels.add(new Option(label.apply(value)));
        return labels;
    }

    private static List<Option> described(SequencedMap<String, String> descriptions) {
        List<Option> options = new ArrayList<>(descriptions.size());
        descriptions.forEach((label, description) -> options.add(new Option(label, description)));
        return options;
    }

    private static void requireText(String text) {
        if (text == null || text.isBlank()) throw new IllegalArgumentException("blank text");
    }

    private static void requireSameSize(List<?> values, List<Option> labels) {
        if (values.size() != labels.size())
            throw new IllegalArgumentException(
                    values.size() + " values and " + labels.size() + " labels");
    }
}
