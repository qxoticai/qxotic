package com.qxotic.jinfer.decision;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * The answers to several questions asked together, each typed by its question. Answers are looked
 * up by the question object that was asked, so keep the instances; a question asked twice in one
 * call holds the last of its answers.
 */
public final class Answers {

    private final Map<Question<?>, Answer> answers = new IdentityHashMap<>();

    Answers() {}

    void put(Question<?> question, float[] probabilities) {
        answers.put(question, answer(question, probabilities));
    }

    /** The answer to {@code question}, which must be one of the questions asked. */
    @SuppressWarnings("unchecked") // filled from (question, its own answer) pairs
    public <A extends Answer> A get(Question<A> question) {
        Answer answer = answers.get(question);
        if (answer == null) throw new IllegalArgumentException("this question was not asked");
        return (A) answer;
    }

    /** {@code probabilities}, in the question's order, as its typed answer. */
    @SuppressWarnings("unchecked") // each case builds the answer type of its question type
    static <A extends Answer> A answer(Question<A> question, float[] probabilities) {
        return (A)
                switch (question) {
                    case Question.Predicate q -> new Answer.Predicate(probabilities[1]); // no, yes
                    case Question.Choice<?> q -> new Answer.Choice<>(q.keys(), list(probabilities));
                    case Question.Rating<?> q ->
                            new Answer.Rating<>(q.levels(), list(probabilities));
                };
    }

    private static List<Float> list(float[] row) {
        List<Float> list = new ArrayList<>(row.length);
        for (float p : row) list.add(p);
        return list;
    }
}
