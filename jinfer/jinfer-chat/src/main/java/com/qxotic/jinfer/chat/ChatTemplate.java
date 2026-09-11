package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.media.Media;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * A model's bidirectional chat codec: conversation to prompt batches, reply tokens to a message.
 */
public interface ChatTemplate {

    /**
     * Tokens written once at the beginning of a new model input sequence. This is model framing,
     * not conversation content; empty means that the model requires no prompt prefix.
     */
    default IntSequence promptStart() {
        return IntSequence.empty();
    }

    /**
     * Streams the complete prompt synchronously. Media chunks are borrowed: neither their liveness
     * nor their contents is guaranteed beyond the sink call ({@link
     * com.qxotic.jinfer.media.MediaProjector#project}), so a sink that ingests after {@code encode}
     * returns must copy them during the call. The returned parser is already seeded with the
     * prompt-owned reply prefix.
     */
    ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink);

    /** As {@link #encode(Conversation, int, Consumer)}, with a caller-owned projected-media LRU. */
    default ReplyState encode(
            Conversation conversation,
            int batchCapacity,
            MediaEncodingCache mediaCache,
            Consumer<Batch> sink) {
        return encode(conversation, batchCapacity, sink);
    }

    /**
     * A fresh, UNSEEDED parser over this family's reply grammar - consulted only when framing fell
     * back to the whole-render (a native {@link #encode} co-produces its seeded parser in {@link
     * ReplyState}). The default is the generic think-span shape every scaffolded reply at least
     * has; a family codec overrides it so the fallback keeps the family's call parsing.
     */
    default ReplyParser parser(Tokenizer tokenizer) {
        return ReplyParser.spans(tokenizer);
    }

    /** The family's grammar-constrained content reply. */
    default Optional<ReplyLanguage.Selection> constrainedReply(String grammar) {
        return Optional.empty();
    }

    /**
     * As {@link #constrainedReply(String)}, with {@code calls} the model may call an offered tool
     * instead of writing the document. Empty when the family has no combined language: the engine
     * then refuses tools together with constrained output.
     */
    default Optional<ReplyLanguage.Selection> constrainedReply(String grammar, boolean calls) {
        return calls ? Optional.empty() : constrainedReply(grammar);
    }

    /**
     * The model's think-span marker spellings - {@link ThinkMarkers#GENERIC} unless the family
     * frames reasoning differently (Gemma 4's {@code <|channel>}/{@code <channel|>} channel span).
     * The sampling policy - masking the markers when thinking is off, capping the span when on -
     * keys on these spellings.
     */
    default ThinkMarkers thinkMarkers() {
        return ThinkMarkers.GENERIC;
    }

    /**
     * What the checkpoint's template can express about reasoning. {@code OPTIONAL} renders a turn
     * with or without a think span; {@code ALWAYS} has no non-thinking turn (LFM2.5-8B-A1B: a bare
     * header the model follows with its own {@code <think>}), so "thinking off" cannot be rendered
     * and the engine refuses it instead of masking the markers, which only turns the reasoning into
     * leaked visible text. Families answer from their template source; the engine reports {@code
     * NONE} when an {@code OPTIONAL} template meets a tokenizer without think markers.
     */
    default ThinkingPolicy thinkingPolicy() {
        return ThinkingPolicy.OPTIONAL;
    }

    /** How a checkpoint reasons: never, when asked, or on every turn. */
    enum ThinkingPolicy {
        NONE,
        OPTIONAL,
        ALWAYS
    }

    /**
     * Default generated-token budget for a reasoning span. Negative leaves it uncapped. Families
     * may override this when their published generation policy expects unrestricted reasoning.
     */
    default int defaultMaxReasoningTokens(int maxOutputTokens) {
        return maxOutputTokens >= 0 ? Math.max(1, maxOutputTokens / 2) : -1;
    }

    /** One family's think-span marker spellings. */
    record ThinkMarkers(String open, String close) {
        /** The {@code <think>}/{@code </think>} convention. */
        static final ThinkMarkers GENERIC = new ThinkMarkers(Thinking.OPEN, Thinking.CLOSE);

        public ThinkMarkers {
            Objects.requireNonNull(open, "open");
            Objects.requireNonNull(close, "close");
        }
    }

    /**
     * Forces the reply to begin a call to one offered tool; argument constraints are
     * family-specific.
     */
    default Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        return Optional.empty();
    }

    /**
     * Best-effort context-position count for one media item - the preprocessing plan's number
     * (image tier, audio frames), never an encoder run. Templates without media keep the default
     * throw; see {@code MediaProjector#positions}.
     */
    default int mediaPositions(Media media) {
        throw new UnsupportedOperationException("this model does not plan media positions");
    }

    record ReplyState(IntSequence replyPrefix, ReplyParser parser) {
        public ReplyState {
            replyPrefix = Objects.requireNonNull(replyPrefix, "replyPrefix");
            parser = Objects.requireNonNull(parser, "parser");
        }
    }
}
