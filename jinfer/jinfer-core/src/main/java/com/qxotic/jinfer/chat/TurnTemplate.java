package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;

import java.util.ArrayList;
import java.util.List;

/** The precise chat contract for curated models: encodes one turn at a time, deterministically and
 *  verbatim. Turn-stability (a turn's batches never change when later turns are appended) is what
 *  makes incremental ingestion and exact prompt caching sound — hand-written implementations
 *  guarantee it by construction, and are validated byte-exact against the model's official Jinja
 *  chat template offline.
 *
 *  <p>Two tokenization domains, enforced by every implementation: turn scaffolding (role headers,
 *  turn and media markers) is emitted as trusted special-token ids; conversation text is tokenized
 *  plainly, so content can never mint control tokens.
 *
 *  <p>{@code instanceof TurnTemplate} is the capability test (same convention as
 *  {@link com.qxotic.jinfer.MultiModal}): templates that implement it get the exact caching path,
 *  plain {@link ChatTemplate}s the best-effort one. */
public interface TurnTemplate extends ChatTemplate {

    /** One turn, lowered to batches. Stateless, deterministic, turn-stable. Conversation-start
     *  tokens (bos) are not a turn's concern — {@link #conversationStart()} owns them. */
    List<Batch> encodeTurn(Message message);

    /** Tokens that open a conversation (bos and any fixed preamble), emitted once before the
     *  first turn. */
    List<Batch> conversationStart();

    /** The assistant generation prefix, appended after the last turn to start decoding.
     *  {@code thinking} toggles the model's reasoning scaffold where one exists (matching the
     *  template's {@code enable_thinking}); models without one ignore it. */
    List<Batch> generationPrompt(boolean thinking);

    /** The assistant turn-close suffix, ingested after a generated reply so the KV ends exactly
     *  where {@link #encodeTurn} of the finished assistant turn would: {@code generationPrompt +
     *  reply tokens + closeTurn} frames identically to {@code encodeTurn(assistant(reply))}. */
    List<Batch> closeTurn();

    /** A whole conversation is its turns, concatenated after the conversation start. */
    @Override
    default List<Batch> encode(List<Message> conversation) {
        List<Batch> out = new ArrayList<>(conversationStart());
        for (Message m : conversation) out.addAll(encodeTurn(m));
        return out;
    }
}
