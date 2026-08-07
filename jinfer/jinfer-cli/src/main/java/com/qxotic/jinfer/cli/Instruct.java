package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JinjaChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** The one-shot {@code --prompt} mode: encode, generate once, exit. The default mode. */
final class Instruct {

    private Instruct() {}

    static <S extends RuntimeState> void run(
            LoadedModel<S> model, ChatTemplate template, Sampler sampler, Options options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        IntSequence promptTokens;
        if (options.rawPrompt()) {
            promptTokens = SpecialTokens.encode(model.tokenizer(), options.prompt());
        } else if (template != null) {
            List<Message> turns = new ArrayList<>();
            if (options.systemPrompt() != null) {
                turns.add(Message.system(options.systemPrompt()));
            }
            turns.add(Message.user(options.prompt()));
            List<Batch> batches =
                    template.encode(new Conversation(turns, List.of(), options.think(), ""));
            promptTokens = IntSequence.wrap(Batch.tokenIds(batches));
        } else {
            List<Object> messages = new ArrayList<>();
            if (options.systemPrompt() != null) {
                messages.add(Map.of("role", "system", "content", options.systemPrompt()));
            }
            messages.add(Map.of("role", "user", "content", options.prompt()));
            promptTokens =
                    new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource())
                            .render(messages, null, true, options.think(), null);
        }
        // --cache / --cache-ro: the prompt cache as a file, through the one facade - which owns
        // the whole policy (codec-less models warn and serve without it, coarse codecs restore
        // read-only, a missing read-only file degrades). Read-write pins the prompt via define()
        // (fine codecs: chunk blocks + a split-last single; coarse: one residue block) and
        // appends the new blocks BEFORE generating - the artifact is the point of --cache, and a
        // generation failure must not lose it. serve() then restores the longest cached prefix
        // (one short, by the law) and generates on top.
        if (options.promptCache() != null && promptTokens.length() >= 2) {
            List<Batch> prompt = List.of(Batch.prefill(promptTokens.toArray()));
            int total = promptTokens.length();
            if (options.echo()) {
                // the whole prompt: serve() ingests it internally, so Turn.generate sees none of it
                Turn.echoPrompt(model.tokenizer(), promptTokens);
            }
            try (PromptCache<S> cache =
                    PromptCache.of(
                            model.model(),
                            model.seed(),
                            PromptCache.Options.DEFAULTS
                                    .withHotSessions(0) // one shot: nothing to keep warm
                                    .withContextCapacity(options.contextCapacity())
                                    .withBlockBudget(Long.MAX_VALUE)
                                    .withCatalog(
                                            options.promptCache(),
                                            options.promptCacheReadOnly()))) {
                if (!options.promptCacheReadOnly() && cache.blockCaching()) {
                    int before = cache.sample().blocks();
                    cache.define(prompt);
                    cache.save();
                    int added = cache.sample().blocks() - before;
                    if (added > 0) {
                        System.err.printf(
                                "cache: %d blocks added, catalog appended (%s)%n",
                                added, options.promptCache());
                    }
                }
                long t0 = System.nanoTime();
                cache.serve(
                        prompt,
                        (state, serving) -> {
                            System.err.printf(
                                    "cache: %d/%d positions restored, prompt ready in %.1f ms%n",
                                    serving.restored(), total, (System.nanoTime() - t0) / 1e6);
                            return Turn.generate(
                                    model,
                                    state,
                                    IntSequence.empty(),
                                    stops,
                                    sampler,
                                    options,
                                    serving::tail);
                        });
            }
            return;
        }

        S state =
                Generator.stateFor(
                        model.model(),
                        promptTokens.length(),
                        options.oneShotCapacity(promptTokens.length()));
        Turn.generate(model, state, promptTokens, stops, sampler, options);
    }
}
