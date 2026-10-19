package com.qxotic.jinfer.server;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Map;

/** OpenAI wire values lowered onto the framework-neutral {@link ChatEngine}. */
final class Generation {

    private final ChatEngine engine;
    private final ServerConfig config;
    private final Metrics metrics;
    private final Sampling defaults;

    Generation(ChatEngine engine, ServerConfig config, Metrics metrics) {
        this.engine = engine;
        this.config = config;
        this.metrics = metrics;
        this.defaults =
                config.defaults().sampling() != null
                        ? config.defaults().sampling()
                        : engine.loaded().samplingDefaults().resolve(null, null, null, null, null);
    }

    PromptCache.Sample cacheSample() {
        return engine.cacheSample();
    }

    MediaEncodingCache.Sample mediaCacheSample() {
        return engine.mediaCacheSample();
    }

    boolean blockCaching() {
        return engine.blockCaching();
    }

    Sampling defaults() {
        return defaults;
    }

    String forcedTool(Map<String, Object> request) {
        return ToolUse.forced(request);
    }

    Reply chat(Map<String, Object> request, List<Object> messages, Sinks sinks) {
        OpenAiChat.Lowered lowered = OpenAiChat.lower(request, messages, config, defaults);
        return recorded(run(lowered, sinks));
    }

    Reply completion(Map<String, Object> request, String prompt, Sinks sinks) {
        IntSequence ids =
                config.defaults().rawPrompt()
                        ? SpecialTokens.encode(engine.loaded().tokenizer(), prompt)
                        : engine.loaded().tokenizer().encode(prompt);
        List<String> stops = OpenAiChat.stops(request.get("stop"));
        try (ChatEngine.Prepared prepared =
                engine.prepareRaw(
                        ids.toArray(),
                        OpenAiChat.sampling(request, defaults),
                        OpenAiChat.maxTokens(request, config),
                        config.limits().requestTimeout(),
                        OpenAiChat.grammar(request),
                        stops)) {
            WireSink sink = new WireSink(request, sinks);
            ChatEngine.Completion completion = engine.complete(prepared, sink);
            sink.finish();
            return recorded(finish(completion, prepared.stops(), sink.inline));
        }
    }

    private Reply run(OpenAiChat.Lowered lowered, Sinks sinks) {
        try (ChatEngine.Prepared prepared = engine.prepare(lowered.request())) {
            WireSink sink = new WireSink(lowered.inlineReasoning(), sinks);
            ChatEngine.Completion completion = engine.complete(prepared, sink);
            sink.finish();
            return finish(completion, prepared.stops(), sink.inline);
        }
    }

    /** The server's one addition to {@link OpenAiChat#reply}: the prompt-cache metric. */
    private Reply finish(
            ChatEngine.Completion completion, List<String> stops, boolean inlineReasoning) {
        Reply reply = OpenAiChat.reply(completion, stops, inlineReasoning);
        metrics.recordPromptCache(completion.tier(), completion.restoredTokens());
        return reply;
    }

    private static final class WireSink implements ChatEngine.ReplySink {
        private final Sinks sinks;
        private final boolean inline;
        private boolean reasoning;

        WireSink(Map<String, Object> request, Sinks sinks) {
            this("none".equals(Values.stringValue(request.get("reasoning_format"), null)), sinks);
        }

        WireSink(boolean inline, Sinks sinks) {
            this.sinks = sinks;
            this.inline = inline;
        }

        @Override
        public void on(ChatEngine.Delta delta) {
            if (delta.channel() == Channel.REASONING) {
                if (inline) {
                    if (!reasoning && sinks.onText() != null) sinks.onText().accept("<think>");
                    reasoning = true;
                    if (sinks.onText() != null) sinks.onText().accept(delta.text());
                } else if (sinks.onReasoning() != null) {
                    sinks.onReasoning().accept(delta.text());
                }
            } else if (delta.channel() == Channel.CONTENT && sinks.onText() != null) {
                if (reasoning) {
                    sinks.onText().accept("</think>");
                    reasoning = false;
                }
                sinks.onText().accept(delta.text());
            }
        }

        void finish() {
            if (inline && reasoning && sinks.onText() != null) sinks.onText().accept("</think>");
        }
    }

    private Reply recorded(Reply reply) {
        metrics.record(reply);
        return reply;
    }
}
