package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.Server;
import java.io.IOException;

/**
 * The {@code --server} mode: the startup banner plus {@link Server#start}. This owns the
 * presentation the server library deliberately has none of: what is being served, where each
 * sampling value came from, and - last, because it is the one line a reader acts on - the endpoint,
 * whose port only exists once {@code start} has bound it.
 *
 * <p>The classpath's full architecture list used to go in the first line, which answered a question
 * nobody asks at startup; it belongs in the "no provider for architecture" error, where it already
 * is.
 */
final class Serve {

    private Serve() {}

    static void run(LoadedModel<?> model, Options options, Sampling sampling) throws IOException {
        // both numbers, because one of them used to be a lie: what this run allocated, and what
        // the model can actually take
        System.out.printf(
                "model       %s  (%s, ctx %d of %d)%n",
                options.modelPath().getFileName(),
                model.model().getClass().getSimpleName(),
                options.contextCapacity(),
                model.model().config().contextLength());
        options.companions()
                .forEach(
                        (capability, file) ->
                                System.out.printf(
                                        "companion   %s = %s%n", capability, file.getFileName()));
        var defaults = model.samplingDefaults();
        System.out.printf(
                "sampling    temperature %s, top-k %s, top-p %s, min-p %s; requests override%n",
                describe(
                        sampling.temperature(),
                        options.temperature() != null,
                        defaults.temperature() != null),
                describe(sampling.topK(), options.topk() != null, defaults.topK() != null),
                describe(sampling.topP(), options.topp() != null, defaults.topP() != null),
                describe(sampling.minP(), options.minp() != null, defaults.minP() != null));
        Server.Running running = Server.start(model, options.toServerConfig(defaults));
        // the CLI never closes the handle; ^C must still free the engine deterministically
        Runtime.getRuntime().addShutdownHook(new Thread(running::close));
        System.out.printf(
                "listening   http://%s:%d  (OpenAI-compatible)%n",
                options.host(), running.address().getPort());
    }

    /**
     * A resolved sampling value with its provenance: the user's flag, the model (GGUF metadata or
     * its port author's recommendation), or jinfer's baseline - so a surprising default explains
     * itself instead of being blamed on the server.
     */
    private static String describe(Number value, boolean userSet, boolean modelRecommended) {
        String source =
                userSet ? "set by user" : modelRecommended ? "model default" : "jinfer default";
        String shown =
                value instanceof Float f
                        ? String.valueOf(Math.round(f * 1000.0) / 1000.0)
                        : value.toString();
        return shown + " (" + source + ")";
    }
}
