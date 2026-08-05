package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Optional;

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
     * Best-effort context-position count for one media item - the preprocessing plan's number
     * (image tier, audio frames), never an encoder run. Templates without media keep the default
     * throw; see {@code Embedder#positions}.
     */
    default int mediaPositions(com.qxotic.jinfer.Media media) {
        throw new UnsupportedOperationException("this model does not plan media positions");
    }

    /**
     * The complete prompt: the framed conversation, ending with the assistant scaffold ({@code
     * conversation.thinking()} toggles the reasoning scaffold where the model has one).
     *
     * @throws UnsupportedConversation when this template cannot frame the conversation byte-exactly
     */
    List<Batch> encode(Conversation conversation);

    /**
     * An encoded prompt together with the trailing ids that grammatically belong to the REPLY -
     * co-produced by {@link #encodePrompt} so the prompt's tail and the parser's initial state can
     * never disagree (the tail may depend on conversation SHAPE, e.g. Gemma opens {@code
     * <|channel>thought\n} only after a trailing tool response with thinking on).
     */
    record Prompt(List<Batch> batches, int[] replySeed) {}

    /**
     * {@link #encode} plus the reply seed the encoded prompt actually ends with. The default
     * derives the seed from the static {@link #replySeed(boolean)} - correct for every template
     * whose tail is a pure function of the thinking flag; a template with a conversation-shaped
     * tail overrides this and builds both in ONE pass.
     */
    default Prompt encodePrompt(Conversation conversation) {
        return new Prompt(encode(conversation), replySeed(conversation.thinking()));
    }

    /**
     * True when this template holds the encoder-projected rows for {@code contentKey} in its
     * in-process media cache, letting a caller who has only the SOURCE bytes skip decoding
     * entirely: pass a keyed {@link Part.Blob} with an EMPTY payload (a frameless {@code
     * Media.Video}) and {@link #encode} replays the cached rows byte-identically to a cold run. A
     * template answering true must honor that empty-payload form - and throw loudly if the entry
     * vanished in between, never encode an empty payload as if it were real. Default: no cache.
     */
    default boolean mediaEncodingCached(byte[] contentKey) {
        return false;
    }

    /**
     * A fresh, single-use parser for one generation pass. Stateful; the driver creates one per
     * request and feeds it every sampled token in order.
     */
    ReplyParser parser();

    /** One-shot decode of a finished reply. */
    default Message decode(IntSequence reply) {
        return ReplyParser.parse(parser(), reply);
    }

    /**
     * The generation prompt's trailing ids that are grammatically part of the REPLY - a think span
     * the template OPENS in the prompt ({@code <think>\n} on Qwen 3.5 / Nemotron / MiniCPM5
     * thinking prompts, or their closed empty pair when thinking is off). The driver pre-feeds them
     * into {@link #parser} so it starts in the right span state; without this, reasoning that
     * begins inside a prompt-opened span routes to the CONTENT channel. Empty when the generation
     * prompt ends at the role scaffold (the model emits its own markers).
     */
    default int[] replySeed(boolean thinking) {
        return new int[0];
    }

    /**
     * Trusted ids seeded into the reply to FORCE a tool call ({@code toolChoice REQUIRED}): the
     * family's call-opening marker ({@code <|tool_call_start|>}, Harmony's {@code <|channel|>}),
     * appended to the prompt so the model can only COMPLETE a call. Empty = this family cannot
     * force calls. The seed deliberately stops BEFORE any delimiter the model's training merges
     * with what follows (never the paren).
     */
    default int[] callSeed() {
        return new int[0];
    }

    /**
     * The dual of {@link #parser} for FORCED tool calls: a GBNF prefix grammar over the family's
     * call syntax - {@code prefix (name|...|name) delimiter} in plain bytes, pinning the reply
     * (already seeded with {@link #callSeed}) to a call of an OFFERED tool. The pin covers only the
     * prefix; once matched the sampler releases and the arguments stay the model's own. Empty when
     * the family has no call syntax (forcing then relies on seeding alone).
     */
    default Optional<String> callGrammar(List<Tool> tools) {
        return Optional.empty();
    }

    /**
     * Trusted scaffold ids that COMPLETE the forced-call header once {@link #callGrammar}'s pin
     * releases - emitted verbatim, never sampled. Empty for families whose header ends at the
     * pinned name (the model's own continuation is on-distribution there); Harmony's header
     * continues {@code " <|constrain|>json<|message|>"} after the name, which is scaffold, not a
     * model choice - improvising it from the pinned (off-policy) state derails generation.
     */
    default int[] callEpilogue() {
        return new int[0];
    }
}
