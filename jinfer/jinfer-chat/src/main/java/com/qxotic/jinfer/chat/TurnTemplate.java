package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * TRANSITIONAL per-turn chat contract for curated models: encodes one turn at a time,
 * deterministically and verbatim. Turn-stability (a turn's batches never change when later turns
 * are appended) is what makes incremental ingestion and exact prompt caching sound — hand-written
 * implementations guarantee it by construction, and are validated byte-exact against the model's
 * official Jinja chat template offline.
 *
 * <p>Two tokenization domains, enforced by every implementation: turn scaffolding (role headers,
 * turn and media markers) is emitted as trusted special-token ids; conversation text is tokenized
 * plainly, so content can never mint control tokens.
 *
 * <p>This is the porting substrate for the {@link ChatTemplate} codec: unported models expose their
 * TurnTemplate through {@link TurnTemplateAdapter}; a native codec port (whole-conversation encode
 * + {@link ReplyParser}) replaces it per model, and the interface goes away with the last port.
 * Per-turn encoding cannot express position-dependent templates (Qwen's last-query rules, Harmony
 * channels) — that is why the codec is whole-conversation.
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
     * Welds the tools and the leading system message into the preamble: appends {@code
     * template.conversationStart(Preamble)} to {@code out} (the whole tool block stays one
     * turn-stable prefix) and returns the remaining turns - the system turn, when consumed by the
     * preamble, is skipped.
     */
    static List<Message> encodePreamble(
            TurnTemplate template, List<Message> messages, List<Tool> tools, List<Batch> out) {
        Optional<Message> system =
                !messages.isEmpty() && messages.get(0).role().equals(Role.SYSTEM)
                        ? Optional.of(messages.get(0))
                        : Optional.empty();
        out.addAll(template.conversationStart(new Preamble(system, tools)));
        return system.isPresent() ? messages.subList(1, messages.size()) : messages;
    }

    /**
     * The preamble including tools: bos plus the model's system/tool framing (for LFM2, one system
     * turn merging the system message and the tool list). Emitted once before the first turn,
     * REPLACING {@link #conversationStart()} AND the leading system turn - the driver must not also
     * encode the system message. The default ignores tools and simply appends the system message as
     * its own turn, which is byte-exact only for templates that do not weld tools into the system
     * turn; a template that welds tools overrides this.
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
     * The conversation as the model's own template would frame it - e.g. a template that
     * unconditionally renders a system turn injects its default here when the conversation lacks
     * one. Identity by default. EVERY caller that encodes turn-by-turn (incremental drivers, the
     * server) must normalize first, or its framing silently drifts from the oracle-validated
     * whole-conversation encoding.
     */
    default List<Message> normalize(List<Message> conversation) {
        return conversation;
    }

    /**
     * The shared codec face for per-turn ports: a plain-text conversation is the normalized turns
     * folded between {@link #conversationStart()} and {@link #generationPrompt}; tools and non-text
     * parts punt to the whole render. Ports whose template welds tools or splices verbatim history
     * (LFM2) override this.
     */
    @Override
    default List<Batch> encode(Conversation conversation) {
        if (!conversation.tools().isEmpty())
            throw new UnsupportedConversation("tool framing not ported: whole-render");
        List<Message> turns = normalize(conversation.messages());
        for (Message m : turns) {
            for (Part part : m.content()) {
                if (!(part instanceof Part.Text))
                    throw new UnsupportedConversation(
                            m.role().name() + " turn: " + part.getClass().getSimpleName());
            }
        }
        List<Batch> out = new ArrayList<>(conversationStart());
        for (Message m : turns) out.addAll(encodeTurn(m));
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }
}
