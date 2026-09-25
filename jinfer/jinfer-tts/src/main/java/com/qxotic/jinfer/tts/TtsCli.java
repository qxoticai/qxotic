package com.qxotic.jinfer.tts;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.AudioPlayer;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class TtsCli {
    private static final String SELF_ARCHIVE = "z://";
    private static final String DEFAULT_ENTRY = "default.gguf";

    private TtsCli() {}

    public static void main(String[] args) {
        try {
            run(Options.parse(args));
        } catch (IllegalArgumentException | UnsupportedOperationException badCommandLine) {
            System.err.println("jinfer-tts: " + badCommandLine.getMessage());
            Options.usage(System.err);
            System.exit(2);
        } catch (AudioPlayer.Failed playerQuit) {
            System.err.println("jinfer-tts: " + playerQuit.getMessage());
            System.exit(playerQuit.status());
        } catch (UncheckedIOException e) {
            System.err.println("jinfer-tts: " + e.getCause().getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("jinfer-tts: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void run(Options options) throws IOException {
        if (options.help()) {
            Options.usage(System.out);
            return;
        }
        boolean needsArchive =
                options.list()
                        || options.model() == null
                        || archived(options.model())
                        || options.companions().values().stream().anyMatch(TtsCli::archived);
        SelfArchive archive =
                needsArchive
                        ? options.archive() == null
                                ? SelfArchive.open()
                                : SelfArchive.open(options.archive())
                        : null;
        if (options.list()) {
            try (archive) {
                System.out.printf("%-48s %10s%n", "entry", "size");
                for (SelfArchive.Entry entry : archive.entries())
                    System.out.printf("%-48s %7.1f MB%n", entry.name(), entry.size() / 1e6);
            }
            return;
        }

        List<Path> extracted = new ArrayList<>();
        // the runtime's best cross-thread arena: ofShared on the JVM, ofAuto in a native image,
        // where a shared arena cannot be closed and Arenas.close is the best-effort release
        Arena arena = Arenas.newCrossThread();
        try (archive) {
            Map<String, Path> companions =
                    resolveCompanions(options.companions(), archive, extracted);
            String source =
                    options.model() == null ? SELF_ARCHIVE + DEFAULT_ENTRY : options.model();
            SpeechSynthesisModel<?, ?, ?> model;
            if (archived(source)) {
                SelfArchive.Entry entry = archive.entry(entryName(source));
                GGUF gguf;
                try (ReadableByteChannel header = archive.channel(entry)) {
                    gguf = GGUF.read(header);
                }
                model =
                        Models.loadSpeech(
                                archive.fileChannel(),
                                gguf,
                                entry.offset(),
                                entry.size(),
                                archive.path(),
                                arena,
                                companions);
            } else {
                model = Models.loadSpeech(Path.of(source), arena, companions);
            }
            use(model, options);
        } finally {
            Arenas.close(arena);
            for (Path path : extracted) Files.deleteIfExists(path);
        }
    }

    private static Map<String, Path> resolveCompanions(
            Map<String, String> sources, SelfArchive archive, List<Path> extracted)
            throws IOException {
        Map<String, Path> resolved = new LinkedHashMap<>();
        for (var companion : sources.entrySet()) {
            String source = companion.getValue();
            Path path;
            if (archived(source)) {
                path = archive.extract(archive.entry(entryName(source)));
                extracted.add(path);
            } else {
                path = Path.of(source);
            }
            resolved.put(companion.getKey(), path);
        }
        return Map.copyOf(resolved);
    }

    private static boolean archived(String source) {
        return source != null && source.startsWith(SELF_ARCHIVE);
    }

    private static String entryName(String source) {
        String name = source.substring(SELF_ARCHIVE.length());
        if (name.isEmpty()) throw new IllegalArgumentException("empty self-archive entry");
        return name;
    }

    private static void use(SpeechSynthesisModel<?, ?, ?> model, Options options)
            throws IOException {
        if (options.stream()) {
            long start = System.nanoTime();
            AudioPlayer.stream(
                    model, options.text(), speechOptions(options), () -> printFirstAudio(start));
        } else if (options.play()) {
            AudioPlayer.play(model.speak(options.text(), speechOptions(options)));
        } else write(model, options);
    }

    private static SpeechOptions speechOptions(Options options) {
        return options.speed() == null ? SpeechOptions.NONE : SpeechOptions.speed(options.speed());
    }

    private static void write(SpeechSynthesisModel<?, ?, ?> model, Options options)
            throws IOException {
        long start = System.nanoTime();
        Media.Audio audio = model.speak(options.text(), speechOptions(options));
        double elapsed = (System.nanoTime() - start) / 1e9;
        Files.write(options.output(), AudioCodec.wav(audio));
        double seconds = audio.pcm().length / (double) audio.channels() / audio.sampleRate();
        System.out.printf(
                "wrote %s%n%.2f s of audio in %.2f s (%.1fx realtime)%n",
                options.output().toAbsolutePath(), seconds, elapsed, seconds / elapsed);
    }

    private static void printFirstAudio(long start) {
        System.out.printf("first audio after %.2f s%n", (System.nanoTime() - start) / 1e9);
    }
}
