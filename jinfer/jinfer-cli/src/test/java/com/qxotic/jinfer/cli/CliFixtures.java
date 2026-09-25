package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.*;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/** Existing weightless model, real engine, borrowed streams: no mocks or model files. */
final class CliFixtures {
    /** Forked CLI checks participate in -Pcoverage instead of disappearing from its report. */
    static List<String> javaCommand() {
        var command =
                new ArrayList<>(
                        List.of(
                                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                                "--add-modules",
                                "jdk.incubator.vector",
                                "--enable-native-access=ALL-UNNAMED"));
        java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments().stream()
                .filter(arg -> arg.startsWith("-javaagent:") && arg.contains("jacoco"))
                .forEach(command::add);
        return command;
    }

    static final class State extends com.qxotic.jinfer.RuntimeState {
        private final Runnable closed;

        State(Runnable closed) {
            this.closed = closed;
        }

        protected void releaseResources() {
            closed.run();
        }
    }

    static final class Capture {
        final ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        final ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        final Main.IO io;
        boolean inputClosed;

        Capture(String input) {
            this(input.getBytes(StandardCharsets.UTF_8));
        }

        Capture(byte[] input) {
            io =
                    new Main.IO(
                            new ByteArrayInputStream(input) {
                                @Override
                                public void close() {
                                    inputClosed = true;
                                }
                            },
                            new PrintStream(stdout, true, StandardCharsets.UTF_8),
                            new PrintStream(stderr, true, StandardCharsets.UTF_8));
        }

        String out() {
            return stdout.toString(StandardCharsets.UTF_8);
        }

        String err() {
            return stderr.toString(StandardCharsets.UTF_8);
        }
    }

    static final class Template implements ChatTemplate {
        final List<Conversation> conversations = new ArrayList<>();
        RuntimeException failure;

        public ReplyState encode(
                Conversation conversation, int capacity, java.util.function.Consumer<Batch> sink) {
            if (failure != null) throw failure;
            if (conversation.messages().getLast().text().equals("reject"))
                throw new IllegalArgumentException("rejected test turn");
            conversations.add(conversation);
            StringBuilder text = new StringBuilder();
            for (Message message : conversation.messages())
                text.append(message.role()).append(':').append(message.text()).append('\n');
            sink.accept(Batch.prefill(TestLanguageModel.TOKENIZER.encode(text).toArray()));
            return new ReplyState(
                    IntSequence.empty(), ReplyParser.spans(TestLanguageModel.TOKENIZER));
        }

        public ThinkingPolicy thinkingPolicy() {
            return ThinkingPolicy.NONE;
        }
    }

    static ChatEngine engine(Template template) {
        var loaded =
                new LoadedModel<>(
                        new TestLanguageModel(),
                        TestLanguageModel.TOKENIZER,
                        "",
                        Set.of(),
                        new ContentKey("cli-test"),
                        Optional.of(template),
                        LoadedModel.SamplingDefaults.NONE);
        return new ChatEngine(loaded, "test", PromptCache.Options.DEFAULTS.withBlockBudget(0));
    }

    private CliFixtures() {}
}
