package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.toknroll.IntSequence;
import java.util.List;

/**
 * A model's chat template as a CODEC with two directions over one grammar: {@link #encode} lowers a
 * {@link Conversation} to the model's token stream, {@link #parser} parses the generated reply
 * stream back into text channels and structure. One implementation per model; {@code
 * JinjaChatTemplate} is the universal whole-render fallback.
 *
 * <p>Encode is a pure function: same conversation, same batches. All prompt-shaping inputs live on
 * the {@link Conversation} (messages, tools, thinking, effort); the returned batches are the
 * COMPLETE prompt, ending with the assistant scaffold, ready to generate. Text lowers to token
 * batches, media to encoder-projected embedding batches. Encoding is whole-conversation (not
 * per-turn) so position-dependent templates (Qwen's last-query rules, Harmony channels) are
 * expressible as local index checks. A conversation shape the template cannot frame byte-exactly
 * throws {@link UnsupportedConversation} - the caller's signal to fall back to the whole-render
 * path.
 *
 * <p>Everything else is a PROPERTY, not API: turn granularity is the prefix-stability law ({@code
 * encode} of a conversation prefix is a token prefix of the extended conversation's encoding, up to
 * the trailing scaffold - the prompt cache and incremental drivers segment by encoding prefixes);
 * batch boundaries are the template's stable cache boundaries (preamble, turns, scaffold last); the
 * String view is {@code tokenizer.decode} of the batches; incremental re-encoding is a
 * longest-common-prefix diff against what a state already ingested.
 *
 * <p>Two tokenization domains, enforced by every implementation: turn scaffolding (role headers,
 * turn and media markers) is emitted as trusted special-token ids; conversation text is tokenized
 * plainly, so content can never mint control tokens. Stop tokens are the model's ({@code
 * stopTokens()}), not the template's; batching policy ({@code Batch.prepare}) is the caller's.
 */
public interface ChatTemplate {

    /**
     * The complete prompt: the framed conversation, ending with the assistant scaffold ({@code
     * conversation.thinking()} toggles the reasoning scaffold where the model has one).
     *
     * @throws UnsupportedConversation when this template cannot frame the conversation byte-exactly
     */
    List<Batch> encode(Conversation conversation);

    /**
     * A fresh, single-use parser for one generation pass. Stateful; the driver creates one per
     * request and feeds it every sampled token in order.
     */
    ReplyParser parser();

    /** One-shot decode of a finished reply. */
    default Message decode(IntSequence reply) {
        return ReplyParser.parse(parser(), reply);
    }
}
