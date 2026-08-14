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
        final class Stop implements Runnable {
            private boolean done;

            @Override
            public synchronized void run() {
                if (done) return;
                running.close();
                engine.savePrompts();
                done = true;
            }
        }
        Stop stop = new Stop();
        Thread shutdown = new Thread(stop, "xjinfer-server-shutdown");
        Runtime.getRuntime().addShutdownHook(shutdown);
        System.out.printf(
                "listening   http://%s:%d  (OpenAI-compatible)%n",
                options.host(), running.address().getPort());
        try {
            running.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            stop.run();
            try {
                Runtime.getRuntime().removeShutdownHook(shutdown);
            } catch (IllegalStateException ignored) {
                // Already shutting down; the hook either performed stop or waited for it.
            }
        }
    }
}
