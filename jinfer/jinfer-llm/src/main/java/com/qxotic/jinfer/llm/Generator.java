package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.boundary.ContextState;
import com.qxotic.jinfer.boundary.LanguageModel;
import com.qxotic.jinfer.telemetry.DecodeEvent;
import java.lang.ref.Reference;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The generation loop: ingest the prompt, then sample-decode-ingest until a stop condition. It
 * knows TOKENS and nothing else - reply structure (think spans, tool calls, UTF-8 assembly) is the
 * chat layer's parser, string-level stops are the chat layer's; both observe this loop through
 * {@link GenerationListener} rather than living inside it.
 */
public final class Generator {

    private Generator() {}

    /**
     * Why a pass ended. {@code TIMEOUT} and {@code LENGTH} are distinct: a wall-clock deadline and
     * a completion budget are operationally different events. Adapters map to coarser wire
     * vocabularies (OpenAI's {@code "length"}) at their edge.
     */
    public enum FinishReason {
        /** A stop token ended the pass. */
        STOP,
        /** The completion budget (or the remaining context) is exhausted. */
        LENGTH,
        /** The wall-clock deadline passed. */
        TIMEOUT,
        /** The listener ended the pass. */
        ABORT
    }

    /**
     * The stopping policy. {@code maxTokens} is the completion budget: {@link #UNLIMITED} means
     * bounded only by the state's remaining context, and 0 is meaningful (prefill without decode).
     * {@code timeout} is the wall-clock budget for the WHOLE pass - prompt ingestion AND decode (a
     * long prefill is exactly the cost a deadline exists to bound); {@link Duration#ZERO} for none.
     * The deadline is cooperative: it is checked BETWEEN steps (a prefill chunk, a decode token), a
     * step in flight always completes, and no new step starts past the deadline - so an expired
     * pass emits nothing it has not already emitted, and no token is ever sampled after it. {@code
     * stopTokens} is the family's terminator set, possibly empty.
     */
    public record Constraints(int maxTokens, Duration timeout, Set<Integer> stopTokens) {

        /** The {@code maxTokens} value for "as much as the context allows". */
        public static final int UNLIMITED = -1;

        public Constraints {
            if (maxTokens < UNLIMITED) {
                throw new IllegalArgumentException("maxTokens " + maxTokens);
            }
            if (timeout == null || timeout.isNegative()) {
                throw new IllegalArgumentException("timeout " + timeout);
            }
            stopTokens = Set.copyOf(stopTokens);
        }
    }

    /**
     * The observer of a pass - ONE object, two lifecycle events, so no intermediary can keep one
     * callback and silently drop the other.
     */
    public interface GenerationListener {

        /**
         * Sees EVERY sampled token in order, the trailing stop token included, before the loop acts
         * on it. Return false to abort: the aborting token is recorded in the result but never
         * ingested into the state ({@link FinishReason#ABORT}).
         */
        boolean onToken(int token);

        /**
         * The token has been COMMITTED to the state - the frontier includes it, which is what
         * per-position cache accounting needs. Never fires for the final token of a pass, which the
         * loop does not ingest.
         */
        default void onIngested(int token) {}
    }

    /**
     * The outcome of one pass. {@code tokens} are the generated ids EXCLUDING the trailing stop
     * token; the array is fresh and ownership transfers to the caller. {@code stopToken} is present
     * exactly when {@code finishReason == STOP}. Times are exact nanoTime deltas: {@code
     * promptTime} is prompt ingestion, {@code decodeTime} is everything after.
     */
    public record GenerationResult(
            int[] tokens,
            OptionalInt stopToken,
            FinishReason finishReason,
            Duration promptTime,
            Duration decodeTime) {

        public int completionTokens() {
            return tokens.length;
        }
    }

    /**
     * As {@link #generate(LanguageModel, ContextState, List, Sampler, Constraints,
     * GenerationListener)} for a plain token prompt.
     */
    public static <S extends ContextState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            int[] promptTokens,
            Sampler sampler,
            Constraints constraints,
            GenerationListener listener) {
        List<Batch> prompt =
                promptTokens.length == 0 ? List.of() : List.of(Batch.prefill(promptTokens));
        return generate(model, state, prompt, sampler, constraints, listener);
    }

    /**
     * One generation pass: ingest {@code prompt} at the state's cursor, then sample-decode-ingest
     * until a stop token, the listener's abort, the deadline, or the budget (clamped to the state's
     * remaining context). A fresh state generates from the prompt; a resumed state (empty prompt)
     * continues from its position. The pass holds the state for its whole duration - a direct
     * {@code ingest} from another thread fails fast instead of silently interleaving two pipelines
     * into one KV. Different STATES of one model may decode concurrently; whether they run in
     * parallel is the backend's business, not this loop's. A deadline that expires DURING prompt
     * ingestion stops at the last completed chunk: the state then holds a partial prompt at its
     * position, and continuing or resetting is the caller's decision.
     */
    public static <S extends ContextState> GenerationResult generate(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            Constraints constraints,
            GenerationListener listener) {
        try {
            return state.exclusively(
                    () -> generate0(model, state, prompt, sampler, constraints, listener));
        } finally {
            Reference.reachabilityFence(model);
        }
    }

    private static <S extends ContextState> GenerationResult generate0(
            LanguageModel<?, ?, S> model,
            S state,
            List<Batch> prompt,
            Sampler sampler,
            Constraints constraints,
            GenerationListener listener) {
        // the STATE's ring is the bound: it may be smaller than what the model was trained for,
        // and it is what the generation actually runs in
        int capacity = state.contextCapacity();
        int promptPositions = state.position() + Batch.positions(prompt);
        if (promptPositions > capacity) {
            throw new IllegalArgumentException(
                    "prompt exceeds context capacity ("
                            + promptPositions
                            + " tokens used, "
                            + capacity
                            + " available - raise the state's contextCapacity)");
        }
        int max =
                constraints.maxTokens() == Constraints.UNLIMITED
                        ? capacity - promptPositions
                        : Math.min(constraints.maxTokens(), capacity - promptPositions);
        long startNanos = System.nanoTime();
        long deadlineNanos =
                constraints.timeout().isZero()
                        ? Long.MAX_VALUE
                        : saturatingDeadline(startNanos, constraints.timeout().toNanos());

        for (Batch batch : Batch.prepare(prompt, state.batchCapacity())) {
            if (System.nanoTime() >= deadlineNanos) {
                break; // no new chunk past the deadline; the decode loop ends the pass below
            }
            model.ingest(state, batch);
        }
        long prefillDoneNanos = System.nanoTime();

        int vocab = model.configuration().vocabularySize();
        Set<Integer> stops = constraints.stopTokens();
        int[] generated = new int[max];
        int n = 0;
        FinishReason finish = FinishReason.LENGTH; // budget/context, or maxTokens=0 prefill-only
        // resolved ONCE, not per token: allocating a per-token event only to find it disabled
        // would let telemetry perturb the thing it measures. A recording started
        // mid-generation is picked up by the next pass, which is soon enough.
        boolean traceTokens = new DecodeEvent().isEnabled();
        while (n < max) {
            // the deadline gates WORK, not delivery: a token sampled in time is always
            // emitted and ingested; no new sample starts past it
            if (System.nanoTime() >= deadlineNanos) {
                finish = FinishReason.TIMEOUT;
                break;
            }
            DecodeEvent decode = traceTokens ? new DecodeEvent() : null;
            if (decode != null) decode.begin();
            try {
                int token = sampler.sampleToken(model.logits(state));
                if (token < 0 || token >= vocab) {
                    throw new IllegalArgumentException(
                            "sampler returned token id "
                                    + token
                                    + " out of range [0, "
                                    + vocab
                                    + ")");
                }
                generated[n++] = token;
                boolean keepGoing = listener.onToken(token);
                if (stops.contains(token)) {
                    finish = FinishReason.STOP;
                    break;
                }
                if (!keepGoing) {
                    finish = FinishReason.ABORT;
                    break;
                }
                if (n >= max || state.position() >= capacity) {
                    break; // LENGTH, and the final token is not ingested
                }
                model.ingest(state, Batch.step(token));
                listener.onIngested(token);
            } finally {
                if (decode != null) {
                    decode.end();
                    decode.commit();
                }
            }
        }
        long endNanos = System.nanoTime();

        if (finish == FinishReason.STOP) {
            return new GenerationResult(
                    Arrays.copyOf(generated, n - 1),
                    OptionalInt.of(generated[n - 1]),
                    finish,
                    Duration.ofNanos(prefillDoneNanos - startNanos),
                    Duration.ofNanos(endNanos - prefillDoneNanos));
        }
        return new GenerationResult(
                Arrays.copyOf(generated, n),
                OptionalInt.empty(),
                finish,
                Duration.ofNanos(prefillDoneNanos - startNanos),
                Duration.ofNanos(endNanos - prefillDoneNanos));
    }

    private static long saturatingDeadline(long nowNanos, long timeoutNanos) {
        return timeoutNanos > Long.MAX_VALUE - nowNanos ? Long.MAX_VALUE : nowNanos + timeoutNanos;
    }
}
