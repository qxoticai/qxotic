package com.qxotic.jinfer.decision;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Typed questions over a {@link DecisionModel}: immutable, safe to share across threads. A call
 * runs in the caller's state, from {@link DecisionModel#newState}, reused from call to call. One
 * question, its answer:
 *
 * <pre>{@code
 * Decisions decisions = Decisions.of(model);
 * try (ContextState state = model.newState(2048)) {
 *     Team team = decisions.ask(state, ticket, Question.choice("Which team?", Team.class))
 *             .selectedOption();
 * }
 * }</pre>
 *
 * Several questions about one input:
 *
 * <pre>{@code
 * var refund = Question.predicate("Does the customer want a refund?");
 * var urgency = Question.rating("How urgent is this?", Urgency.class);
 * Answers answers = decisions.ask(state, ticket, refund, urgency);
 * answers.get(refund).isYes();
 * answers.get(urgency).selectedLevel();
 * }</pre>
 *
 * {@link #ask(ContextState, String, Question...)} takes them the model's way, with no isolation
 * guaranteed; {@link #askIndependently} answers each as if asked alone.
 *
 * <p>One call at a time per state; open one state per thread over the same model to run side by
 * side.
 */
public final class Decisions {

    private final DecisionModel model;

    private Decisions(DecisionModel model) {
        this.model = model;
    }

    /** Typed questions over {@code model}. */
    public static Decisions of(DecisionModel model) {
        return new Decisions(Objects.requireNonNull(model, "model"));
    }

    /** Asks one question about {@code input}. */
    public <A extends Answer> A ask(ContextState state, String input, Question<A> question) {
        return ask(state, text(input), question);
    }

    /** As {@link #ask(ContextState, String, Question)} about text and media parts, in order. */
    public <A extends Answer> A ask(ContextState state, List<Content> input, Question<A> question) {
        require(state, input);
        Objects.requireNonNull(question, "question");
        return Answers.answer(question, model.decide(state, input, lower(question)));
    }

    /**
     * Asks the questions about {@code input}, each answer found by its question, the way the model
     * was trained to take several questions. No isolation is guaranteed: a model trained to read
     * them together reads the input once for all of them, and an answer can then shift with the
     * questions asked beside it and with their order. Use {@link #askIndependently} when it must
     * not.
     */
    public Answers ask(ContextState state, String input, Question<?>... questions) {
        return ask(state, text(input), questions);
    }

    /** As {@link #ask(ContextState, String, Question...)} about text and media parts, in order. */
    public Answers ask(ContextState state, List<Content> input, Question<?>... questions) {
        require(state, input);
        return new Answers(questions, model.decide(state, input, lower(questions)));
    }

    /**
     * Asks every question about {@code input} independently of the others: each answer is the one
     * {@link #ask(ContextState, String, Question)} gives for that question alone, found by its
     * question.
     */
    public Answers askIndependently(ContextState state, String input, Question<?>... questions) {
        return askIndependently(state, text(input), questions);
    }

    /**
     * As {@link #askIndependently(ContextState, String, Question...)} about text and media parts,
     * in order.
     */
    public Answers askIndependently(
            ContextState state, List<Content> input, Question<?>... questions) {
        require(state, input);
        return new Answers(questions, model.decideIndependently(state, input, lower(questions)));
    }

    private static List<Content> text(String input) {
        return List.of(new Content.Text(Objects.requireNonNull(input, "input")));
    }

    // a caller's null is named here; anything else is the model's to refuse
    private static void require(ContextState state, List<Content> input) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(input, "input");
    }

    private static List<DecisionModel.Question> lower(Question<?>[] questions) {
        Objects.requireNonNull(questions, "questions");
        List<DecisionModel.Question> lowered = new ArrayList<>(questions.length);
        for (Question<?> question : questions)
            lowered.add(lower(Objects.requireNonNull(question, "question")));
        return lowered;
    }

    /** The question as the model takes it. */
    private static DecisionModel.Question lower(Question<?> question) {
        return switch (question) {
            case Question.Predicate q ->
                    new DecisionModel.Predicate(q.text(), q.yesMeans(), q.noMeans());
            case Question.Choice<?> q -> new DecisionModel.Choice(q.text(), q.labels());
            case Question.Rating<?> q -> new DecisionModel.Rating(q.text(), q.labels());
        };
    }
}
