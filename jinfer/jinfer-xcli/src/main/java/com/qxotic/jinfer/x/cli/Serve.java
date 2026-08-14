package com.qxotic.jinfer.x.cli;

import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.llm.Sampling;
import com.qxotic.jinfer.x.server.Server;
import java.io.IOException;

/** Presentation and process lifetime for {@code --server}; the server library stays silent. */
final class Serve {

    private Serve() {}

    static void run(ChatEngine engine, LoadedModel<?> model, Sampling sampling, Options options)
            throws IOException {
        System.out.printf(
                "model       %s  (%s, ctx %d of %d)%n",
                options.modelPath().getFileName(),
                model.model().getClass().getSimpleName(),
                options.contextCapacity(),
                model.model().config().contextLength());
        System.out.printf(
                "speculation %s (depth %d)%n",
                engine.speculationReady() ? "ready" : "unavailable", engine.speculationDepth());
        Server.Running running = Server.start(engine, options.serverConfig(sampling));
        Runtime.getRuntime().addShutdownHook(new Thread(running::close, "xjinfer-server-shutdown"));
        System.out.printf(
                "listening   http://%s:%d  (OpenAI-compatible)%n",
                options.host(), running.address().getPort());
        try {
            running.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            running.close();
        }
    }
}
