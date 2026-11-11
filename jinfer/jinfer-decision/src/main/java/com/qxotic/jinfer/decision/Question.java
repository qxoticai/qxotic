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
        return choice(text, List.of(options), option -> option);
    }

    /** Pick one option by label, each with its description. */
    static Choice<String> choice(String text, SequencedMap<String, String> descriptions) {
        return new Choice<>(text, List.copyOf(descriptions.keySet()), described(descriptions));
    }

    /** Pick one of {@code keys}, whatever the caller acts on, each shown by its {@code label}. */
    static <E> Choice<E> choice(String text, List<E> keys, Function<? super E, String> label) {
        return new Choice<>(text, keys, labelled(keys, label));
    }

    /** Where on the scale of {@code levels}, lowest first, the input falls. */
    static Rating<String> rating(String text, String... levels) {
        List<String> keys = List.of(levels);
        return new Rating<>(text, keys, labelled(keys, level -> level));
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
     * Pick one of {@code keys}. Order is part of the question: it is the order the model reads the
     * options in.
     *
     * @param <E> what the caller acts on: an enum constant, a tool, a document
     * @param options beside the keys: how the model reads each one
     */
    record Choice<E>(String text, List<E> keys, List<Option> options)
            implements Question<Answer.Choice<E>> {
        public Choice {
            requireText(text);
            keys = List.copyOf(keys);
            options = List.copyOf(options);
            requireSameSize(keys, options);
            if (keys.size() < 2)
                throw new IllegalArgumentException("a choice needs at least 2 options");
        }
    }

    /**
     * Where on the scale of {@code levels}, lowest first.
     *
     * @param <E> what the caller reads a level as: a string, an enum constant
     * @param options beside the levels: how the model reads each one
     */
    record Rating<E>(String text, List<E> levels, List<Option> options)
            implements Question<Answer.Rating<E>> {
        public Rating {
            requireText(text);
            levels = List.copyOf(levels);
            options = List.copyOf(options);
            requireSameSize(levels, options);
            if (levels.size() < 2)
                throw new IllegalArgumentException("a rating needs at least 2 levels");
        }
    }

    private static String readable(Enum<?> constant) {
        // a leading or trailing underscore would otherwise reach the model as a stray space
        return constant.name().toLowerCase(Locale.ROOT).replace('_', ' ').strip();
    }

    private static <E> List<Option> labelled(List<E> keys, Function<? super E, String> label) {
        List<Option> options = new ArrayList<>(keys.size());
        for (E key : keys) options.add(new Option(label.apply(key)));
        return options;
    }

    private static List<Option> described(SequencedMap<String, String> descriptions) {
        List<Option> options = new ArrayList<>(descriptions.size());
        descriptions.forEach((label, description) -> options.add(new Option(label, description)));
        return options;
    }

    private static void requireText(String text) {
        if (text == null || text.isBlank()) throw new IllegalArgumentException("blank text");
    }

    private static void requireSameSize(List<?> keys, List<Option> options) {
        if (keys.size() != options.size())
            throw new IllegalArgumentException(
                    keys.size() + " keys and " + options.size() + " options");
    }
}
