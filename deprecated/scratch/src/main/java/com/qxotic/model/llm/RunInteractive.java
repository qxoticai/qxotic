package com.qxotic.model.llm;

import com.qxotic.model.llm.generic.ChatMLFormat;
import com.qxotic.model.llm.llama.Llama;
import com.qxotic.model.llm.llama.TraceDebug;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

public class RunInteractive {

    public static final int BATCH_SIZE = Integer.getInteger("com.qxotic.BatchSize", 16);

    public static <Configuration, Weights, State> void runInteractive(
            Model<Configuration, Weights, State> model,
            Weights weights,
            ChatFormat chatFormat,
            Sampler<State> sampler,
            Options options) {
        int batchSize = BATCH_SIZE;
        State state = null;
        IntSequence.Builder conversationTokens = IntSequence.newBuilder();
        chatFormat.beginOfText().ifPresent(conversationTokens::add);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(
                    chatFormat.encodeMessage(
                            new ChatFormat.Message(ChatMLFormat.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            if (List.of("quit", "exit").contains(userText)) {
                break;
            }
            if (state == null) {
                state = model.createNewState(batchSize);
            }
            conversationTokens.addAll(
                    chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(ChatFormat.ASSISTANT));
            Set<Integer> stopTokens = chatFormat.stopTokens();

            IntConsumer onPromptToken =
                    promptToken -> {
                        if (options.echo()) {
                            System.err.print(chatFormat.echo(IntSequence.of(promptToken)));
                        }
                    };

            IntPredicate onGeneratedToken =
                    token -> {
                        if (options.echo()) {
                            System.err.print(chatFormat.echo(IntSequence.of(token)));
                        }
                        if (options.stream()) {
                            System.out.print(chatFormat.stream(IntSequence.of(token)));
                        }
                        return !stopTokens.contains(
                                token); // continue, unless a stop token is found
                    };

            IntSequence responseTokens =
                    generateTokens(
                            model,
                            weights,
                            state,
                            batchSize,
                            startPosition,
                            conversationTokens
                                    .asSequenceView()
                                    .subSequence(startPosition, conversationTokens.size()),
                            options.maxTokens(),
                            sampler,
                            onPromptToken,
                            onGeneratedToken);

            // Include stop token in the conversation history, but not in the response displayed to
            // the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens = responseTokens.subSequence(0, responseTokens.length() - 1);
            }
            if (!options.stream()) {
                String responseText = chatFormat.echo(responseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model model to run inference (including weights, configuration, tokenizer ...)
     * @param state state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition start prompt ingestion + inference at this position in the context e.g.
     *     useful if state was kept across calls (chained generation). 0 implies run with no
     *     previous context.
     * @param promptTokens prompt tokens to ingest, all the prompt tokens will be ingested, given
     *     there's enough capacity left in the context
     * @param maxTokens maximum number of tokens (can go up to {@link
     *     Llama.Configuration#contextLength context length} if this value is negative or greater
     *     than {@link Llama.Configuration#contextLength context length}
     * @param sampler {@link Sampler strategy} used to select tokens
     * @param onGeneratedToken callback, if non-null, it's called every time a token is inferred
     *     e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not
     *     include any token from the prompt
     */
    public static <Configuration, Weights, State> IntSequence generateTokens(
            Model<Configuration, Weights, State> model,
            Weights weights,
            State state,
            int maxBatchSize,
            int startPosition,
            IntSequence promptTokens,
            int maxTokens,
            Sampler<State> sampler,
            IntConsumer onPromptToken,
            IntPredicate onGeneratedToken) {
        long startAllNanos = System.nanoTime();
        IntSequence.Builder generatedTokens = IntSequence.newBuilder();

        int promptIndex = 0;
        int position = 0;
        for (position = startPosition;
                promptIndex < promptTokens.length() && position < maxTokens;
                ++position) {
            int nTokens =
                    Math.min(
                            maxTokens - position,
                            Math.min(promptTokens.length() - promptIndex, maxBatchSize));
            nTokens =
                    Integer.highestOneBit(
                            nTokens); // batches should be <= batchSize and powers of 2.

            final int[] tokens = new int[nTokens];
            for (int i = 0; i < nTokens; i++) {
                tokens[i] = promptTokens.intAt(promptIndex + i);
                TraceDebug.token("ingest", position + i, tokens[i]);
            }
            // Force-pick token from prompt.
            model.ingestTokens(weights, state, tokens);
            if (onPromptToken != null) {
                for (int promptToken : tokens) {
                    onPromptToken.accept(promptToken);
                }
            }

            promptIndex += nTokens;
            position += nTokens - 1; // incremented in the loop
        }

        long startGenNanos = System.nanoTime();
        for (; position < maxTokens; ++position) {
            // Compute logits from previous token.
            model.computeLogits(weights, state);
            int generatedToken = sampler.applyAsInt(state);
            TraceDebug.token("produce", position, generatedToken);
            // Ingest generated token.
            model.ingestTokens(weights, state, new int[] {generatedToken});
            generatedTokens.add(generatedToken);
            if (onGeneratedToken != null) {
                if (!onGeneratedToken.test(generatedToken)) {
                    break;
                }
            }
        }

        int promptTokensCount = promptIndex;
        int generatedTokensCount = generatedTokens.size();

        long nowNanos = System.nanoTime();
        long promptNanos = startGenNanos - startAllNanos;
        long genNanos = nowNanos - startGenNanos;
        System.err.printf(
                "%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                position,
                maxTokens,
                promptTokensCount / (promptNanos / 1_000_000_000.0),
                promptTokensCount,
                generatedTokensCount / (genNanos / 1_000_000_000.0),
                generatedTokensCount);

        return generatedTokens.build();
    }
}
