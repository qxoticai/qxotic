package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Encoded audio or live 16 kHz mono s16le PCM to text. Only live raw input uses the transcription
 * HUD; redirected output remains a plain transcript.
 */
final class Transcribe {

    private Transcribe() {}

    static final class Settings {
        boolean rawPcm;
        TranscriptHud.Theme theme = TranscriptHud.Theme.BUNDLED.getFirst();
    }

    static boolean read(Options o, Options.Args a) {
        switch (a.name) {
            case "--raw-pcm" -> o.transcription.rawPcm = a.flag();
            case "--theme" -> {
                String value = a.value();
                o.transcription.theme = TranscriptHud.Theme.named(value);
                Options.require(
                        o.transcription.theme != null,
                        "--theme must be one of %s; got '%s'",
                        TranscriptHud.Theme.names(),
                        value);
            }
            default -> {
                return false;
            }
        }
        o.use(a.name, Set.of("transcribe"));
        return true;
    }

    static void validate(Options o) {
        Options.require(
                o.input != null && !o.input.isBlank(),
                "transcribe requires an audio file or '-' for stdin");
        if (o.legacy && o.input.equals("-")) o.transcription.rawPcm = true;
        Options.require(
                !o.transcription.rawPcm || o.input.equals("-"), "--raw-pcm requires '-' for stdin");
        Options.require(
                !o.supplied("--theme") || o.transcription.rawPcm,
                "--theme applies only to live --raw-pcm input");
    }

    static int run(Options options, Main.IO io, ModelStore store) throws IOException {
        Media.Audio audio = null;
        if (!options.transcription.rawPcm) {
            boolean stdin = options.input.equals("-");
            byte[] encoded = stdin ? io.read("audio") : null;
            try {
                audio =
                        stdin
                                ? AudioCodec.decode(encoded)
                                : AudioCodec.load(Path.of(options.input));
            } catch (IOException | IllegalArgumentException e) {
                throw Main.failure(
                        "cannot decode audio " + (stdin ? "from stdin" : "'" + options.input + "'"),
                        e);
            }
        }
        Options.Files files = options.resolve(store);
        Arena arena = Arenas.newCrossThread();
        try {
            TranscriptionModel<?, ?, ?> model;
            try (var spinner = LoadSpinner.start("Loading model", io)) {
                model = Models.loadTranscription(files.model(), arena, files.companions());
                if (options.transcription.rawPcm
                        && System.getProperty("org.graalvm.nativeimage.imagecode") == null)
                    warmUp(model);
            } catch (IOException | IllegalArgumentException | UnsupportedOperationException e) {
                throw Main.failure("cannot prepare transcription model '" + files.model() + "'", e);
            }
            execute(model, audio, options, io);
            return Thread.currentThread().isInterrupted() ? 130 : 0;
        } finally {
            Arenas.close(arena);
        }
    }

    static void execute(
            TranscriptionModel<?, ?, ?> model, Media.Audio audio, Options options, Main.IO io)
            throws IOException {
        Terminal terminal =
                options.transcription.rawPcm && io.isTerminal(2)
                        ? Terminal.stderr(io.err(), options.color)
                        : null;
        Transcription result =
                options.transcription.rawPcm
                        ? pump(model, io.in(), terminal, options.transcription.theme, io.err())
                        : model.transcribe(audio);
        // The interactive view has already settled the transcript; redirected stdout always gets
        // it.
        if (terminal == null || !io.isTerminal(1) || result.tokens().isEmpty())
            io.out().println(result.text());
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer transcribe - turn audio into text
                Usage: jinfer [model options] transcribe [options] <audio|->
                Examples:
                  jinfer transcribe -m parakeet.gguf recording.wav
                  jinfer transcribe -m parakeet.gguf - < recording.wav

                  --raw-pcm                  live stdin: 16 kHz mono signed 16-bit little-endian PCM
                  --theme <name>             live-view palette: mint, nord, catppuccin, ember, frost, mono

                '-' reads encoded audio unless --raw-pcm is supplied. Legacy --transcribe - remains raw PCM.
                Final transcripts go to stdout; live partials and progress go to stderr.
                """);
        Options.modelHelp(out);
    }

    /**
     * On the JVM the first decodes run cold while the JIT compiles them, seconds on a busy machine;
     * paid here, behind the load spinner, they never hold up live audio.
     */
    private static <S extends RuntimeState> void warmUp(TranscriptionModel<?, ?, S> model) {
        try (S state = model.newState();
                TranscriptionStream stream = model.stream(state)) {
            stream.feed(new float[5 * model.sampleRate()]); // past a chunk commit
            stream.partial();
        }
    }

    /**
     * Streams stdin PCM through the model, into the live view on {@code terminal}, else the plain
     * log. Final pieces show as soon as they commit; the partial refreshes after every half second
     * of new audio. Drain captured input before asking for another partial.
     */
    private static <S extends RuntimeState> Transcription pump(
            TranscriptionModel<?, ?, S> model,
            InputStream in,
            Terminal terminal,
            TranscriptHud.Theme theme,
            PrintStream err)
            throws IOException {
        int rate = model.sampleRate();
        if (rate != 16000)
            throw new IllegalArgumentException(
                    "--raw-pcm requires a model accepting 16000 Hz audio");
        TranscriptHud view =
                terminal == null ? null : new TranscriptHud(terminal, Terminal::columns, theme);
        int refreshEvery = rate / 2; // samples of new audio per partial
        // Decoding pauses for partials and chunk commits, but a live source cannot: if this
        // thread stops reading, the pipe backs up and the capture side drops microphone audio,
        // which reaches the model as spliced garbage. A reader thread keeps stdin drained; the
        // queue absorbs decode bursts. Bounded, so a decode that cannot keep up at all
        // backpressures like any pipe instead of buffering without limit.
        BlockingQueue<float[]> queue = new ArrayBlockingQueue<>(8192); // ~17 min of 0.125 s chunks
        float[] eof = new float[0];
        IOException[] readFailure = new IOException[1];
        Thread reader =
                new Thread(
                        () -> {
                            byte[] bytes = new byte[4000]; // 0.125 s of s16le at 16 kHz
                            try {
                                int read;
                                while ((read = in.readNBytes(bytes, 0, bytes.length)) > 0) {
                                    if ((read & 1) != 0)
                                        throw new IOException(
                                                "truncated s16le PCM: odd trailing byte");
                                    int samples = read / 2;
                                    float[] pcm = new float[samples];
                                    for (int i = 0; i < samples; i++) {
                                        int lo = bytes[2 * i] & 0xFF, hi = bytes[2 * i + 1];
                                        pcm[i] = ((short) ((hi << 8) | lo)) / 32768f;
                                    }
                                    if (view != null) view.level(rms(pcm));
                                    queue.put(pcm);
                                }
                            } catch (IOException e) {
                                readFailure[0] = Main.failure("cannot read raw PCM from stdin", e);
                            } catch (InterruptedException interrupted) {
                                Thread.currentThread().interrupt();
                            } finally {
                                try {
                                    queue.put(eof);
                                } catch (InterruptedException interrupted) {
                                    Thread.currentThread().interrupt();
                                }
                            }
                        },
                        "jinfer-stdin-reader");
        reader.setDaemon(true);
        reader.start();
        try (S state = model.newState();
                TranscriptionStream stream = model.stream(state)) {
            StringBuilder text = new StringBuilder(); // the final pieces so far
            List<Transcription.Token> tokens = new ArrayList<>();
            List<Transcription.Token> tail = List.of();
            long fed = 0, fresh = 0; // samples fed in total, and since the last partial
            int logged = 0; // characters of the final text the plain log has printed
            while (true) {
                float[] pcm = queue.take();
                // feed everything captured so far before spending time on a partial
                while (pcm != eof) {
                    Transcription piece = stream.feed(pcm);
                    fed += pcm.length;
                    fresh += pcm.length;
                    if (!piece.text().isEmpty() || !piece.tokens().isEmpty()) {
                        text.append(piece.text());
                        tokens.addAll(piece.tokens());
                        if (view != null) view.show(tokens, tail);
                        else logged = logWords(text, logged, false, err);
                    }
                    pcm = queue.poll();
                    if (pcm == null) break;
                }
                if (pcm == eof) break;
                if (fresh >= refreshEvery) {
                    fresh = 0;
                    Transcription partial = stream.partial();
                    tail = partial.tokens();
                    if (view != null) view.show(tokens, tail);
                    else err.printf("… %ds %s%n", fed / rate, partial.text().strip());
                }
            }
            if (readFailure[0] != null) throw readFailure[0];
            Transcription last = stream.finish();
            text.append(last.text());
            tokens.addAll(last.tokens());
            Transcription finished = new Transcription(text.toString(), tokens);
            if (view != null) view.finish(finished.words());
            else logWords(text, logged, true, err);
            return finished;
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new InterruptedIOException("transcription interrupted");
        } finally {
            reader.interrupt();
            // Borrowed stdin cannot be closed here. Interruptible sources stop immediately;
            // ponytail: native stdin may remain blocked until EOF/process exit, so its reader is a
            // daemon.
            try {
                reader.join(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (view != null) view.close();
        }
    }

    /** RMS about the mean, so a microphone's DC offset does not read as a constant level. */
    private static float rms(float[] pcm) {
        if (pcm.length == 0) return 0;
        double mean = 0, sum = 0;
        for (float sample : pcm) mean += sample;
        mean /= pcm.length;
        for (float sample : pcm) sum += (sample - mean) * (sample - mean);
        return (float) Math.sqrt(sum / pcm.length);
    }

    /**
     * Logs the final text past {@code logged} on a line of its own, up to the last word the next
     * piece might still continue unless {@code all}; returns how far it has logged.
     */
    private static int logWords(StringBuilder text, int logged, boolean all, PrintStream err) {
        int end = all ? text.length() : text.lastIndexOf(" "); // a word starts at its space
        if (end <= logged) return logged;
        String words = text.substring(logged, end).strip();
        if (!words.isEmpty()) err.println(words);
        return end;
    }
}
