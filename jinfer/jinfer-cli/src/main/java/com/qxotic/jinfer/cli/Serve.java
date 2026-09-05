package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.Server;
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
                engine.contextCapacity(),
                model.model().configuration().contextLength());
        System.out.printf(
                "speculation %s (depth %d)%n",
                engine.speculationReady() ? "ready" : "unavailable", engine.speculationDepth());
        var config = options.serverConfig(sampling);
        Server.Running running;
        try {
            running = Server.start(engine, config);
        } catch (java.net.BindException e) {
            throw new IllegalArgumentException(
                    "port "
                            + config.bind().getPort()
                            + " on "
                            + config.bind().getHostString()
                            + " is already in use (--port picks another)",
                    e);
        }
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
        Thread shutdown = new Thread(stop, "jinfer-server-shutdown");
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
