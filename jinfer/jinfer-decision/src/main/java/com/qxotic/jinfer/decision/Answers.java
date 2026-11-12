package com.qxotic.jinfer.decision;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The answers to several questions, each typed by its question: immutable, safe to share across
 * threads. An answer is found by its question or any equal one; a question asked twice in one call
 * holds the last of its answers.
 */
public final class Answers {

    private final Map<Question<?>, Answer> answers = new HashMap<>();

    Answers(Question<?>[] questions, float[][] rows) {
        for (int i = 0; i < questions.length; i++)
            answers.put(questions[i], answer(questions[i], rows[i]));
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
                    case Question.Choice<?> q ->
                            new Answer.Choice<>(q.options(), list(probabilities));
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
