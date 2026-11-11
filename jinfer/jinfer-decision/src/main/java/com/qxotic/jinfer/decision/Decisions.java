package com.qxotic.jinfer.decision;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Typed questions over a {@link DecisionModel}, with the state the model runs in kept here. One
 * question, its answer:
 *
 * <pre>{@code
 * try (Decisions decisions = Decisions.over(model, 2048)) {
 *     Team team = decisions.ask(ticket, Question.choice("Which team?", Team.class))
 *             .selectedOption();
 * }
 * }</pre>
 *
 * Several questions about one input:
 *
 * <pre>{@code
 * var refund = Question.predicate("Does the customer want a refund?");
 * var urgency = Question.rating("How urgent is this?", Urgency.class);
 * Answers answers = decisions.askEach(ticket, refund, urgency);
 * answers.get(refund).isYes();
 * answers.get(urgency).selectedLevel();
 * }</pre>
 *
 * {@link #askEach} answers every question as if it were asked alone. {@link #askTogether} lets a
 * model trained for it read the input once for all of them, much faster; the questions are then not
 * kept apart, and an answer can shift with what is asked beside it.
 *
 * <p>One call runs at a time; open one {@code Decisions} per thread over the same model to run side
 * by side.
 */
public final class Decisions implements AutoCloseable {

    private final DecisionModel model;
    private final ContextState state;

    private Decisions(DecisionModel model, ContextState state) {
        this.model = model;
        this.state = state;
    }

    /** Typed questions over {@code model}, at its default capacity: 4096 tokens. */
    public static Decisions over(DecisionModel model) {
        return over(model, 0);
    }

    /**
     * Typed questions over {@code model}, for inputs that take up to {@code contextCapacity} tokens
     * together with their questions; 0 is the default, -1 the model's maximum. The state's memory
     * grows with the capacity: size it to the inputs.
     */
    public static Decisions over(DecisionModel model, int contextCapacity) {
        Objects.requireNonNull(model, "model");
        return new Decisions(model, model.newState(contextCapacity));
    }

    /** Asks one question about {@code input}. */
    public <A extends Answer> A ask(String input, Question<A> question) {
        return ask(text(input), question);
    }

    /** As {@link #ask(String, Question)} about text and media parts, in order. */
    public synchronized <A extends Answer> A ask(List<Content> input, Question<A> question) {
        return Answers.answer(question, model.decide(state, input, lower(question)));
    }

    /**
     * Asks every question about {@code input} on its own: each answer is the one {@link #ask}
     * gives, found by its question.
     */
    public Answers askEach(String input, Question<?>... questions) {
        return askEach(text(input), questions);
    }

    /** As {@link #askEach(String, Question...)} about text and media parts, in order. */
    public synchronized Answers askEach(List<Content> input, Question<?>... questions) {
        return answers(questions, model.decideEach(state, input, lower(questions)));
    }

    /**
     * Asks the questions about {@code input} together, each answer found by its question. A model
     * trained to reads the input once for all of them; an answer can then shift with the questions
     * asked beside it and with their order.
     */
    public Answers askTogether(String input, Question<?>... questions) {
        return askTogether(text(input), questions);
    }

    /** As {@link #askTogether(String, Question...)} about text and media parts, in order. */
    public synchronized Answers askTogether(List<Content> input, Question<?>... questions) {
        return answers(questions, model.decideTogether(state, input, lower(questions)));
    }

    private static List<Content> text(String input) {
        return List.of(new Content.Text(input));
    }

    @Override
    public synchronized void close() {
        state.close();
    }

    private static Answers answers(Question<?>[] questions, float[][] rows) {
        Answers answers = new Answers();
        for (int i = 0; i < questions.length; i++) answers.put(questions[i], rows[i]);
        return answers;
    }

    private static List<DecisionModel.Question> lower(Question<?>[] questions) {
        List<DecisionModel.Question> lowered = new ArrayList<>(questions.length);
        for (Question<?> question : questions) lowered.add(lower(question));
        return lowered;
    }

    /** The question as the model takes it. */
    private static DecisionModel.Question lower(Question<?> question) {
        return switch (question) {
            case Question.Predicate q ->
                    new DecisionModel.Predicate(q.text(), q.yesMeans(), q.noMeans());
            case Question.Choice<?> q -> new DecisionModel.Choice(q.text(), q.options());
            case Question.Rating<?> q -> new DecisionModel.Rating(q.text(), q.options());
        };
    }
}
