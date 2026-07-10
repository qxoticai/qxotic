package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The precise chat contract for curated models: encodes one turn at a time, deterministically and
 * verbatim. Turn-stability (a turn's batches never change when later turns are appended) is what
 * makes incremental ingestion and exact prompt caching sound — hand-written implementations
 * guarantee it by construction, and are validated byte-exact against the model's official Jinja
 * chat template offline.
 *
 * <p>Two tokenization domains, enforced by every implementation: turn scaffolding (role headers,
 * turn and media markers) is emitted as trusted special-token ids; conversation text is tokenized
 * plainly, so content can never mint control tokens.
 *
 * <p>Discovery is {@code model.turnTemplate()} — the Optional capability accessor on {@code
 * LanguageModel} is THE mechanism for finding a model's template (models without one fall back to
 * whole-render). {@code instanceof} is only for refining an already-obtained {@link ChatTemplate}
 * to its per-turn form.
 */
public interface TurnTemplate extends ChatTemplate {

    /**
     * One turn, lowered to batches. Stateless, deterministic, turn-stable. Conversation-start
     * tokens (bos) are not a turn's concern — {@link #conversationStart()} owns them.
     */
    List<Batch> encodeTurn(Message message);

    /**
     * Tokens that open a conversation (bos and any fixed preamble), emitted once before the first
     * turn.
     */
    List<Batch> conversationStart();

    /**
     * The conversation-scoped, turn-stable inputs the preamble is built from: the leading system
     * message (if any) and the offered tools. Both are fixed for the life of the conversation, so
     * the preamble stays a stable cache prefix as turns are appended.
     */
    record Preamble(Optional<Message> system, List<Tool> tools) {
        public Preamble {
            tools = List.copyOf(tools);
        }
    }

    /**
     * Whether this template can frame tool definitions and tool-call/result turns byte-exactly with
     * the model's own chat template. Only models whose tool block is a stable prefix (tools in the
     * system/developer preamble, not injected before the last user turn) should return true; the
     * rest fall back to whole-render. Default false.
     */
    default boolean supportsTools() {
        return false;
    }

    /**
     * The preamble including tools: bos plus the model's system/tool framing (for LFM2, one system
     * turn merging the system message and the tool list). Emitted once before the first turn,
     * REPLACING {@link #conversationStart()} AND the leading system turn - the driver must not also
     * encode the system message. The default ignores tools and simply appends the system message as
     * its own turn, which is byte-exact only for templates that do not weld tools into the system
     * turn; a {@link #supportsTools()} template overrides this.
     */
    default List<Batch> conversationStart(Preamble preamble) {
        List<Batch> out = new ArrayList<>(conversationStart());
        preamble.system().ifPresent(sys -> out.addAll(encodeTurn(sys)));
        return out;
    }

    /**
     * The assistant generation prefix, appended after the last turn to start decoding. {@code
     * thinking} toggles the model's reasoning scaffold where one exists (matching the template's
     * {@code enable_thinking}); models without one ignore it.
     */
    List<Batch> generationPrompt(boolean thinking);

    /**
     * The assistant turn-close suffix, ingested after a generated reply so the KV ends exactly
     * where {@link #encodeTurn} of the finished assistant turn would: {@code generationPrompt +
     * reply tokens + closeTurn} frames identically to {@code encodeTurn(assistant(reply))}.
     */
    List<Batch> closeTurn();

    /**
     * A fresh detector that turns this model's generated token ids into structured tool calls, or
     * empty when the model has no native tool-call format. The detector is stateful and single-use;
     * the generation driver creates one per request. This is the decode-side counterpart to the
     * (encode-side) tool rendering, and it is per-model because the call format is - see {@link
     * ToolCallDetector}.
     */
    default Optional<ToolCallDetector> toolCallDetector() {
        return Optional.empty();
    }

    /**
     * The conversation as the model's own template would frame it - e.g. a template that
     * unconditionally renders a system turn injects its default here when the conversation lacks
     * one. Identity by default. EVERY caller that encodes turn-by-turn (incremental drivers, the
     * server) must normalize first, or its framing silently drifts from the oracle-validated
     * whole-conversation encoding.
     */
    default List<Message> normalize(List<Message> conversation) {
        return conversation;
    }

    /** A whole conversation is its normalized turns, concatenated after the conversation start. */
    @Override
    default List<Batch> encode(List<Message> conversation) {
        List<Batch> out = new ArrayList<>(conversationStart());
        for (Message m : normalize(conversation)) out.addAll(encodeTurn(m));
        return out;
    }
}
