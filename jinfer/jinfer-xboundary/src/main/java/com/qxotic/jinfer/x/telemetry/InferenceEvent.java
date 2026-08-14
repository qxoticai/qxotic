package com.qxotic.jinfer.x.telemetry;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Threshold;
import jdk.jfr.Timespan;

/**
 * One model call - chat, embedding, rerank or speech. Emitted by jinfer; consume it through JFR
 * ({@code jfr print --events jinfer.Inference}, JMC, or a {@code RecordingStream}) rather than
 * constructing it.
 *
 * <p>The field vocabulary is OpenTelemetry's {@code gen_ai.*}, so an exporter is a rename table:
 *
 * <table border="1">
 *   <caption>OpenTelemetry mapping</caption>
 *   <tr><th>field</th><th>attribute</th></tr>
 *   <tr><td>{@code model}</td><td>{@code gen_ai.request.model}, {@code gen_ai.response.model}</td></tr>
 *   <tr><td>{@code operation}</td><td>{@code gen_ai.operation.name}</td></tr>
 *   <tr><td>{@code outputType}</td><td>{@code gen_ai.output.type}</td></tr>
 *   <tr><td>{@code inputTokens}</td><td>{@code gen_ai.usage.input_tokens}</td></tr>
 *   <tr><td>{@code outputTokens}</td><td>{@code gen_ai.usage.output_tokens}</td></tr>
 *   <tr><td>{@code reasoningTokens}</td><td>{@code gen_ai.usage.reasoning.output_tokens}</td></tr>
 *   <tr><td>{@code cachedTokens}</td><td>{@code gen_ai.usage.cache_read.input_tokens}</td></tr>
 *   <tr><td>{@code finishReason}</td><td>{@code gen_ai.response.finish_reasons} (one, not a list)</td></tr>
 *   <tr><td>{@code errorType}</td><td>{@code error.type}</td></tr>
 *   <tr><td>{@code duration}</td><td>derives {@code gen_ai.client.operation.duration}</td></tr>
 * </table>
 *
 * <p>{@code prefillTime}/{@code decodeTime} are jinfer's own: OpenTelemetry's time-to-first-token
 * and time-per-output-token are DERIVED from them, and the split says more than either. Zero is a
 * true measurement everywhere - an embedding runs no decode loop, so its {@code decodeTime} is
 * genuinely zero rather than "not applicable".
 *
 * <p>{@code gen_ai.provider.name} and {@code gen_ai.response.model} are deliberately absent: in
 * process they are constants, so an exporter adds them for free and every event would pay bytes to
 * repeat them.
 */
@Name("jinfer.Inference")
@Label("Inference")
@Category({"jinfer", "Inference"})
@Description("One model call: chat, embedding, rerank or speech.")
@StackTrace(false)
@Threshold("0 ms")
public final class InferenceEvent extends Event {

    /** Chat: a conversation. Embedding: one call, batched or not. */
    public static final String CHAT = "chat";

    /** OpenTelemetry spells embeddings in the plural. */
    public static final String EMBEDDINGS = "embeddings";

    /** No OpenTelemetry value fits a cross-encoder score, so this is jinfer's own. */
    public static final String RERANK = "rerank";

    /** Speech is {@code generate_content} with {@link #SPEECH} output, not its own operation. */
    public static final String GENERATE_CONTENT = "generate_content";

    public static final String TEXT = "text";
    public static final String JSON = "json";
    public static final String SPEECH = "speech";

    /**
     * A started event with the identity fields set and the string fields defaulted to empty rather
     * than null - every emission site goes through here.
     *
     * <p>There are several such sites (chat, embeddings, the server's own generation seam, a
     * rejected request) and hand-filling them let the defaults drift once already: a rejected
     * request reported a null {@code outputType} and {@code finishReason} while every other path
     * reported a string. A consumer filtering on those fields silently missed a whole class of
     * event. Fill identity and defaults in one place; callers add only what they measured.
     */
    public static InferenceEvent started(String model, String operation, String outputType) {
        InferenceEvent event = new InferenceEvent();
        event.model = model;
        event.operation = operation;
        event.outputType = outputType;
        event.finishReason = "";
        event.errorType = "";
        event.cacheTier = "";
        event.begin();
        return event;
    }

    @Label("Model")
    public String model;

    @Label("Operation")
    public String operation;

    @Label("Output Type")
    public String outputType;

    @Label("Input Tokens")
    public int inputTokens;

    @Label("Output Tokens")
    public int outputTokens;

    @Label("Reasoning Tokens")
    public int reasoningTokens;

    @Label("Cached Tokens")
    public int cachedTokens;

    /**
     * Which source served the prompt: {@code session} (append-only reuse, nothing restored), {@code
     * blocks} (a restore from the block tree), or {@code fresh}. Cut from an earlier draft as
     * "server internals" - wrongly: the two reuse paths cost very differently, and the ratio is
     * what tells you whether the retained-session limit is set high enough.
     */
    @Label("Cache Tier")
    public String cacheTier;

    @Label("Queue Time")
    @Timespan(Timespan.NANOSECONDS)
    public long queueTime;

    @Label("Prefill Time")
    @Timespan(Timespan.NANOSECONDS)
    public long prefillTime;

    @Label("Decode Time")
    @Timespan(Timespan.NANOSECONDS)
    public long decodeTime;

    @Label("Finish Reason")
    public String finishReason;

    /**
     * Empty on success. Low cardinality by contract - a class name or a fixed slug, never a
     * message.
     */
    @Label("Error Type")
    public String errorType;
}
