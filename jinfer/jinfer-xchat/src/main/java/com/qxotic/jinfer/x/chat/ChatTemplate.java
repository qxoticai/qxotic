package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Media;
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
     * Streams the complete prompt synchronously. Media chunks are borrowed: neither their liveness
     * nor their contents is guaranteed beyond the sink call ({@link
     * com.qxotic.jinfer.x.boundary.Embedder#embed}), so a sink that ingests after {@code encode}
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

    /** The family's output grammar composed with calls to the offered tools. */
    default Optional<ReplyLanguage.Selection> constrainedReply(
            String contentGbnf, List<Tool> callableTools) {
        return Optional.empty();
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
     * throw; see {@code Embedder#positions}.
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
