package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * The per-turn chat contract for curated models: encodes one turn at a time, deterministically and
 * verbatim. Turn-stability (a turn's batches never change when later turns are appended) is what
 * makes incremental ingestion and exact prompt caching sound — hand-written implementations
 * guarantee it by construction, and are validated byte-exact against the model's official Jinja
 * chat template offline. This is the PERMANENT substrate native ports are written against; the
 * default {@link #encode(Conversation)} below is the {@link ChatTemplate} codec face the oracles
 * validate.
 *
 * <p>Two tokenization domains, enforced by the seam itself: turn scaffolding (role headers, turn
 * and media markers) is emitted as trusted special-token ids (specials-only lookups); conversation
 * text goes through the plain, non-special-aware tokenizer path, so content can never mint control
 * tokens.
 *
 * <p>Position-dependent framing (Qwen's last-query rules, Harmony channels) and tool welding are
 * not expressible turn-locally — a port with those overrides {@link #encode(Conversation)} directly
 * (LFM2 does) while keeping its per-turn methods as the building blocks.
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
     * The part shapes a text-only tool codec frames byte-exactly - text anywhere, calls and
     * reasoning on assistant turns, results on tool turns; anything else (media, misplaced parts)
     * throws so the caller falls back to the whole render. The shared validator behind every
     * tool-capable port's {@code encode} override.
     */
    /**
     * The prompt-opened think scaffold shared by qwen-style families: {@code <think>\n} when
     * thinking, the closed empty pair {@code <think>\n\n</think>\n\n} when not - the exact ids the
     * template appends after the role header, and therefore the {@code replySeed}.
     */
    static int[] reasonSeed(Tokenizer tokenizer, boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(SpecialTokens.require(tokenizer, Thinking.OPEN));
        if (thinking) {
            ids.addAll(tokenizer.encode("\n"));
        } else {
            ids.addAll(tokenizer.encode("\n\n"));
            ids.add(SpecialTokens.require(tokenizer, Thinking.CLOSE));
            ids.addAll(tokenizer.encode("\n\n"));
        }
        return ids.build().toArray();
    }

    static void requireToolShapes(List<Message> messages) {
        for (Message m : messages) {
            boolean assistant = m.role().equals(Role.ASSISTANT);
            boolean tool = m.role().equals(Role.TOOL);
            for (Part p : m.content()) {
                boolean ok =
                        p instanceof Part.Text
                                || (assistant
                                        && (p instanceof Part.ToolCall
                                                || p instanceof Part.Reasoning))
                                || (tool && p instanceof Part.ToolResult);
                if (!ok)
                    throw new UnsupportedConversation(
                            m.role().name() + " turn: " + p.getClass().getSimpleName());
            }
        }
    }

    /**
     * The conversation as the model's own template would frame it - e.g. a template that
     * unconditionally renders a system turn (Llama) injects its default here when the conversation
     * lacks one. Identity by default. EVERY caller that encodes turn-by-turn must normalize first,
     * or its framing silently drifts from the oracle-validated whole-conversation encoding.
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
