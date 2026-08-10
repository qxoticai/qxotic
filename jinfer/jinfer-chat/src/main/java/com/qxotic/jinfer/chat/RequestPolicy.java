package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The shared request policy: the standard sampling stack, channel-scoped grammar constraint and the
 * forced-call recipe, as statics over {@link LoadedModel} - jinfer-server (which never builds a
 * {@link ChatEngine}) and both framework providers share exactly this code.
 */
public final class RequestPolicy {

    private RequestPolicy() {}

    /**
     * The id to EMIT when a decode must be ended from outside (a grammar's dead end, a forced
     * call's terminator): the stop set's FIRST element - the model's own end-of-turn, an order
     * {@link SpecialTokens#stops} establishes and {@code LoadedModel} preserves.
     */
    private static int endTurn(LoadedModel<?> m) {
        return m.stopTokens().iterator().next();
    }

    /**
     * The smallest completion budget a think span fits: below it, thinking is disabled per request
     * regardless of the model default - silently spending a tiny budget on reasoning scaffold would
     * return empty visible text.
     */
    public static final int THINK_FLOOR = 16;

    /**
     * The standard jinfer sampling stack: a resolved {@link Sampling} plus the reasoning policy -
     * thinking on caps the think span so it cannot starve the visible answer ({@code
     * reasoningOverride}: null = half of {@code maxTokens}, -1 = uncapped); thinking off masks the
     * think markers outright.
     */
    public static Sampler sampler(
            LoadedModel<?> m,
            Sampling sampling,
            boolean think,
            int maxTokens,
            Integer reasoningOverride,
            int[] replySeed) {
        Sampler sampler = sampling.sampler(m.model().config().vocabularySize());
        if (!think) {
            return Thinking.banMarkers(sampler, m.tokenizer());
        }
        int budget =
                reasoningOverride != null
                        ? reasoningOverride
                        : maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1;
        // prompt-opened spans (replySeed carries the open id): the cap must start ARMED - the
        // open token never passes through the sampler on those families. The seed is the
        // ENCODED prompt's tail ({@link ChatTemplate.Prompt}); it may depend on conversation
        // shape, which is why the caller supplies it instead of this method querying the
        // template's static answer.
        boolean startInThink = false;
        OptionalInt open = SpecialTokens.find(m.tokenizer(), Thinking.OPEN);
        if (open.isPresent()) {
            for (int t : replySeed) {
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
            LoadedModel<?> m, Sampler s, Grammar.Cursor g, int[] replySeed) {
        // channel-scoped: the model's OWN reply parser is the channel authority - the grammar
        // exists only where text becomes output (reasoning, scaffold and call payloads stay
        // free; Harmony's analysis channel reasons at full strength under a JSON schema). The
        // wrapper owns a private parser copy pre-fed the reply seed (the encoded prompt's
        // tail), so it starts in the exact span state the prompt left the model in.
        ReplyParser parser =
                m.template()
                        .map(ChatTemplate::parser)
                        .orElseGet(() -> ReplyParser.spans(m.tokenizer()));
        for (int t : replySeed) {
            parser.feed(t);
        }
        Set<String> output = parser.outputChannels();
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
                endTurn(m));
    }

    /**
     * Tools AND a schema-constrained answer as ONE selection: the family's auto language with the
     * content hole carrying {@code contentGbnf} - the model may call its tools in its own syntax
     * (thinking stays free), and visible text can only be the schema. Empty when the family
     * declares no {@link ChatTemplate#autoLanguage} - the caller's cue to reject the combination.
     */
    public static Optional<Sampler> toolsWithSchema(
            LoadedModel<?> m, String contentGbnf, Sampler base, int[] replySeed) {
        return m.template()
                .flatMap(t -> t.autoLanguage(ReplyLanguage.gbnf(contentGbnf)))
                .map(
                        language -> {
                            ReplyLanguage.Walk walk =
                                    ReplyLanguage.Selection.of(language, m.tokenizer()).walk();
                            for (int t : replySeed) walk.feed(t);
                            walk.beginReply();
                            return walk.sampler(base, endTurn(m));
                        });
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
    public static Optional<ForcedCall> forceCall(
            LoadedModel<?> m, List<Tool> tools, Sampler base, int[] replySeed) {
        ChatTemplate template = m.template().orElse(null);
        if (template == null) return Optional.empty();
        // the reply-language path: ONE walk constrains the whole call - header, an OFFERED name,
        // SCHEMA-BOUND arguments - leaving no free region to derail in (the legacy pin releases
        // after the name and the arguments were the model's own, the recorded defect class)
        Optional<ReplyLanguage.Node> language = template.forcedCallLanguage(tools);
        if (language.isPresent()) {
            ReplyLanguage.Selection sel = ReplyLanguage.Selection.of(language.get(), m.tokenizer());
            int[] seed = sel.forcedPrefix();
            ReplyLanguage.Walk walk = sel.walk();
            for (int t : seed) walk.feed(t);
            Sampler constrained = walk.sampler(base, endTurn(m));
            int[] parserSeed = new int[replySeed.length + seed.length];
            System.arraycopy(replySeed, 0, parserSeed, 0, replySeed.length);
            System.arraycopy(seed, 0, parserSeed, replySeed.length, seed.length);
            return Optional.of(new ForcedCall(Batch.prefill(seed), constrained, parserSeed));
        }
        if (template.callSeed().length == 0) return Optional.empty();
        int[] callSeed = template.callSeed();
        Sampler sampler = base;
        Optional<String> pin = template.callGrammar(tools);
        if (pin.isPresent()) {
            sampler =
                    Sampler.withPrefixGrammar(
                            base,
                            Grammar.of(pin.get(), m.tokenizer()).cursor(),
                            endTurn(m),
                            template.callEpilogue());
        }
        // the caller's seed is the encoded prompt's own tail; a forced render is thinking-off,
        // so for every shipped template it is empty or the closed non-thinking scaffold
        int[] parserSeed = new int[replySeed.length + callSeed.length];
        System.arraycopy(replySeed, 0, parserSeed, 0, replySeed.length);
        System.arraycopy(callSeed, 0, parserSeed, replySeed.length, callSeed.length);
        return Optional.of(new ForcedCall(Batch.prefill(callSeed), sampler, parserSeed));
    }

    /**
     * States {@code schema} to the model, appended to the last user message - the other half of a
     * schema-constrained request.
     *
     * <p>A grammar decides what the reply may LOOK like; only the prompt says what it must MEAN. A
     * schema applied as a grammar alone yields well-formed nonsense: asked to extract a person from
     * "Johann is 42", a model that never saw the schema answers {@code {"name":"user_agent","age":
     * 42}} - valid under the grammar, wrong in every way that matters. Hosted providers hide this
     * because their models are trained on a schema channel; a local GGUF has no such channel, so
     * the schema must be said out loud.
     *
     * <p>Appended to the LAST user message rather than added as a message of its own: it is that
     * request's instruction, it keeps a cached prefix's bytes untouched, and it is where
     * langchain4j itself puts the schema when a provider declares no schema support - so a jinfer
     * prompt reads like every other provider's. ponytail: on a MULTI-TURN structured conversation
     * the statement moves to each round's newest user turn, so the previous round's bytes diverge
     * and the warm-session tier re-ingests - a latency ceiling, never a correctness one; a
     * schema-scoped placement would fix it if extraction loops ever get long.
     *
     * <p>{@code toolsOffered} switches to the composed wording - "and nothing else" talks a model
     * out of CALLING its tools first, so with tools present the statement binds the eventual
     * answer, not the whole reply.
     *
     * <p>Returns {@code messages} unchanged when {@code schema} is null/empty or no user message is
     * present (a schema stated to nobody would be a silent prompt mutation).
     */
    public static List<Message> stating(
            List<Message> messages, Map<String, Object> schema, boolean toolsOffered) {
        if (schema == null || schema.isEmpty()) return messages;
        int last = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i).role() == Role.USER) {
                last = i;
                break;
            }
        }
        if (last < 0) return messages;
        List<Message> out = new ArrayList<>(messages);
        Message user = out.get(last);
        List<Part> content = new ArrayList<>(user.content());
        content.add(new Part.Text(statement(schema, toolsOffered)));
        out.set(last, new Message(user.role(), content));
        return out;
    }

    /**
     * {@link #stating} over the OpenAI-shaped maps a Jinja whole-render fallback consumes - same
     * statement, same placement, so BOTH encode paths of one request state the schema identically
     * (the fallback serves unported models, exactly the ones a grammar-only schema fails worst on).
     * Maps whose last user {@code content} is not a plain string return unchanged, like a
     * conversation with no user message.
     */
    public static List<Object> statingMaps(List<Object> maps, Map<String, Object> schema) {
        if (schema == null || schema.isEmpty()) return maps;
        for (int i = maps.size() - 1; i >= 0; i--) {
            if (!(maps.get(i) instanceof Map<?, ?> m) || !"user".equals(m.get("role"))) continue;
            // the LAST user turn decides: a non-string content there returns unchanged rather
            // than smuggling the statement onto an EARLIER turn
            if (!(m.get("content") instanceof String content)) return maps;
            @SuppressWarnings("unchecked")
            var user = new LinkedHashMap<>((Map<String, Object>) m);
            user.put("content", content + statement(schema, false));
            List<Object> out = new ArrayList<>(maps);
            out.set(i, user);
            return out;
        }
        return maps;
    }

    /** The one statement both {@code stating} shapes append. */
    private static String statement(Map<String, Object> schema, boolean toolsOffered) {
        return (toolsOffered
                        ? "\nWhen you can answer, reply with JSON matching this schema, and"
                                + " nothing else:\n"
                        : "\nYou must answer with JSON matching this schema, and nothing else:\n")
                + JsonCodec.stringify(schema);
    }
}
