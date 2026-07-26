package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.util.List;
import java.util.Optional;

/**
 * The shared request policy: the standard sampling stack, channel-scoped grammar constraint and the
 * forced-call recipe, as statics over {@link LoadedModel} - jinfer-server (which never builds a
 * {@link ChatEngine}) and both framework providers share exactly this code.
 */
public final class RequestPolicy {

    private RequestPolicy() {}

    /**
     * The smallest completion budget a think span fits: below it, thinking is disabled per request
     * regardless of the model default - silently spending a tiny budget on reasoning scaffold would
     * return empty visible text.
     */
    public static final int THINK_FLOOR = 16;

    /**
     * The standard jinfer sampling stack: (temperature, topP, seed) plus the reasoning policy -
     * thinking on caps the think span so it cannot starve the visible answer ({@code
     * reasoningOverride}: null = half of {@code maxTokens}, -1 = uncapped); thinking off masks the
     * think markers outright.
     */
    public static Sampler sampler(
            LoadedModel<?> m,
            float temperature,
            float topP,
            long seed,
            boolean think,
            int maxTokens,
            Integer reasoningOverride) {
        Sampler sampler =
                Sampler.select(m.model().config().vocabularySize(), temperature, topP, seed);
        if (!think) {
            return Thinking.banMarkers(sampler, m.tokenizer());
        }
        int budget =
                reasoningOverride != null
                        ? reasoningOverride
                        : maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1;
        // prompt-opened spans (replySeed carries the open id): the cap must start ARMED - the
        // open token never passes through the sampler on those families
        boolean startInThink = false;
        java.util.OptionalInt open = SpecialTokens.find(m.tokenizer(), Thinking.OPEN);
        if (open.isPresent() && m.template().isPresent()) {
            for (int t : m.template().get().replySeed(think)) {
                if (t == open.getAsInt()) {
                    startInThink = true;
                    break;
                }
            }
        }
        return Thinking.capBudget(sampler, m.tokenizer(), budget, startInThink);
    }

    /**
     * Grammar-constrained decoding over any compiled cursor (JSON mode, JSON schema, raw GBNF):
     * dormant through the think span (constraining from token 0 would suppress reasoning), the
     * newline after {@code </think>} passed through, and a dead-end forcing one of the model's own
     * stop tokens.
     */
    public static Sampler constrained(
            LoadedModel<?> m, Sampler s, Grammar.Cursor g, boolean think) {
        // channel-scoped: the model's OWN reply parser is the channel authority - the grammar
        // exists only where text becomes output (reasoning, scaffold and call payloads stay
        // free; Harmony's analysis channel reasons at full strength under a JSON schema). The
        // wrapper owns a private parser copy pre-fed the reply seed, so it starts in the exact
        // span state the prompt left the model in.
        ReplyParser parser =
                m.template()
                        .map(ChatTemplate::parser)
                        .orElseGet(() -> ReplyParser.spans(m.tokenizer()));
        for (int t : m.template().map(tp -> tp.replySeed(think)).orElseGet(() -> new int[0])) {
            parser.feed(t);
        }
        java.util.Set<String> output = parser.outputChannels();
        var tokenizer = m.tokenizer();
        // the pre-start escape: only the span-OPENING marker (the model's right to reason) -
        // never stop/turn specials, which would let the model end the turn instead of complying
        int[] escape = SpecialTokens.find(tokenizer, Thinking.OPEN).stream().toArray();
        return new ChannelConstrainedSampler(
                s,
                parser,
                channel -> output.contains(channel) ? g : null,
                token -> SpecialTokens.isSpecial(tokenizer, token),
                escape,
                SpecialTokens.newlineTokens(tokenizer),
                m.stopTokens().iterator().next());
    }

    /**
     * The complete forced-call recipe as ONE value - the three parts only work together, so a
     * caller can never seed without pre-feeding the parser (the historical bug class): {@code seed}
     * joins the prompt (the model can only COMPLETE a call), {@code sampler} prefix-pins the
     * offered names and forces the family's header epilogue before releasing, {@code parserSeed}
     * puts the reply parser in the seeded span state. A forced reply never thinks (it is seeded
     * INTO the call block), so the prompt must be rendered with thinking off.
     */
    public record ForcedCall(Batch seed, Sampler sampler, int[] parserSeed) {}

    /**
     * The recipe for forcing a call to one of {@code tools}, or empty when this model cannot force
     * (no native codec, or its template declares no call seed) - the caller's cue to reject.
     */
    public static Optional<ForcedCall> forceCall(LoadedModel<?> m, List<Tool> tools, Sampler base) {
        ChatTemplate template = m.template().orElse(null);
        if (template == null || template.callSeed().length == 0) return Optional.empty();
        int[] callSeed = template.callSeed();
        Sampler sampler = base;
        Optional<String> pin = template.callGrammar(tools);
        if (pin.isPresent()) {
            sampler =
                    Sampler.withPrefixGrammar(
                            base,
                            Grammar.of(pin.get(), m.tokenizer()).cursor(),
                            m.stopTokens().iterator().next(),
                            template.callEpilogue());
        }
        int[] replySeed = template.replySeed(false);
        int[] parserSeed = new int[replySeed.length + callSeed.length];
        System.arraycopy(replySeed, 0, parserSeed, 0, replySeed.length);
        System.arraycopy(callSeed, 0, parserSeed, replySeed.length, callSeed.length);
        return Optional.of(new ForcedCall(Batch.prefill(callSeed), sampler, parserSeed));
    }
}
