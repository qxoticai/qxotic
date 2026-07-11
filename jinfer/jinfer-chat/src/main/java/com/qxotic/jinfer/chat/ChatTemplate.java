package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.toknroll.IntSequence;
import java.util.List;

/**
 * A model's chat template as a CODEC with two directions over one grammar: {@link #encode} lowers a
 * {@link Conversation} to the model's token stream, {@link #decoder} parses the generated reply
 * stream back into structured {@link Part}s. One implementation per model.
 *
 * <p>Encode is a pure function: same conversation, same batches. All prompt-shaping inputs live on
 * the {@link Conversation} (messages, tools, thinking, effort); the returned batches are the
 * COMPLETE prompt, ending with the assistant scaffold, ready to generate. Text lowers to {@link
 * Batch.Input.Tokens}, media to encoder-projected {@link Batch.Input.Embeddings}. Encoding is
 * whole-conversation (not per-turn) so position-dependent templates (Qwen's last-query rules,
 * Harmony channels) are expressible as local index checks.
 *
 * <p>Everything else is a PROPERTY, not API: turn granularity is the prefix-stability law ({@code
 * encode} of a conversation prefix is a token prefix of the extended conversation's encoding, up to
 * the trailing scaffold - the prompt cache and incremental drivers segment by encoding prefixes);
 * the String view is {@code tokenizer.decode} of the batches; incremental re-encoding is a
 * longest-common-prefix diff against what a state already ingested.
 *
 * <p>Two tokenization domains, enforced by every implementation: turn scaffolding (role headers,
 * turn and media markers) is emitted as trusted special-token ids; conversation text is tokenized
 * plainly, so content can never mint control tokens. Stop tokens are the model's ({@code
 * stopTokens()}), not the template's; batching policy ({@code Batch.prepare}) is the caller's.
 */
public interface ChatTemplate {

    /**
     * Whether this template can frame {@code conversation} byte-exactly with the model's own
     * reference template - the caller's routing question (an unsupported conversation falls back to
     * the whole-render path). Checks structure (part kinds, tool framing), never content.
     */
    boolean supports(Conversation conversation);

    /**
     * The complete prompt: the framed conversation, ending with the assistant scaffold ({@code
     * conversation.thinking()} toggles the reasoning scaffold where the model has one). Batch
     * boundaries are the template's stable cache boundaries (preamble, turns, scaffold last).
     */
    List<Batch> encode(Conversation conversation);

    /**
     * A fresh, single-use decoder for one generation pass. Stateful; the driver creates one per
     * request and feeds it every sampled token in order.
     */
    ReplyDecoder decoder();

    /** One-shot decode of a finished reply (trailing stop token excluded). */
    default Message decode(IntSequence reply) {
        ReplyDecoder decoder = decoder();
        reply.forEachInt(decoder::feed);
        decoder.finish();
        return decoder.message();
    }
}
