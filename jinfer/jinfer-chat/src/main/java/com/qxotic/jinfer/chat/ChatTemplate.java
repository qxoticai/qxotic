package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
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
    default int mediaPositions(Media media) {
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
     * The COMPILED forced-call selection over {@code tools} - per-tool call regions with
     * SCHEMA-BOUND arguments, so a forced call can neither name an unoffered tool nor malform its
     * payload (the free region after a released pin was one defect class: LFM2.5's hallucinated
     * argument, Mistral's post-pin derail, gpt-oss's malformed JSON). When present, {@code
     * RequestPolicy.forceCall} drives the whole forced reply through the selection's walk. Empty =
     * this family cannot force calls.
     *
     * <p>HERMETIC on purpose, like {@link #constrainedAuto}: the template compiles its own tree
     * with its own tokenizer; the {@link ReplyLanguage.Node} authoring vocabulary is the template
     * AUTHOR'S currency and never crosses this interface.
     */
    default Optional<ReplyLanguage.Selection> forcedCall(List<Tool> tools) {
        return Optional.empty();
    }

    /**
     * The family's compiled constrained selection: visible CONTENT can only be {@code contentGbnf}
     * and thinking stays free. {@code toolsOffered} states the reply's rights - true admits the
     * family's own calls and the answer is optional (the model may call instead); false REQUIRES
     * the document and admits no calls (an empty reply must not comply). SOURCE in, the COMPILED
     * selection out - the {@link ReplyLanguage.Node} authoring vocabulary never crosses this seam.
     * Empty = this family has no reply language; the caller falls back or rejects.
     */
    default Optional<ReplyLanguage.Selection> constrainedAuto(
            String contentGbnf, boolean toolsOffered) {
        return Optional.empty();
    }
}
