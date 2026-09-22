package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * {@code --transcribe <audio|->}: the transcript of an audio file, or with {@code -}, live
 * transcription of raw 16 kHz mono s16le PCM streamed on stdin. On a terminal, stderr shows the
 * live view ({@link TranscriptHud}); redirected, or on a terminal that cannot move the cursor, it
 * logs the final text in whole words and each partial on lines of their own. The final transcript
 * alone goes to stdout, so it pipes. Pipe a microphone in with ffmpeg, then {@code ... -ar 16000
 * -ac 1 -f s16le - | jinfer -m parakeet.gguf --transcribe -}, capturing with {@code -f pulse -i
 * default} on Linux, {@code -f avfoundation -i ":0"} on macOS, or {@code -f dshow -i
 * audio="Microphone"} on Windows (cmd, or PowerShell 7.4 and on, which pipe bytes unchanged).
 */
final class Transcribe {

    private Transcribe() {}

    static int run(Options options) {
        boolean stdin = "-".equals(options.transcribeAudio().toString());
        Media.Audio audio = null;
        if (!stdin) {
            try {
                audio = AudioCodec.load(options.transcribeAudio());
            } catch (IOException | IllegalArgumentException e) {
                System.err.println(
                        "ERROR cannot decode "
                                + options.transcribeAudio()
                                + ": "
                                + Options.rootMessage(e));
                return 1;
            }
        }
        LoadSpinner spinner = LoadSpinner.start("Loading model");
        // newCrossThread, not ofShared: a native image degrades to an ofAuto arena it cannot close
        Arena arena = Arenas.newCrossThread();
        try {
            TranscriptionModel<?, ?, ?> model;
            try {
                model = Models.loadTranscription(options.modelPath(), arena);
            } catch (IllegalArgumentException
                    | IllegalStateException
                    | UnsupportedOperationException
                    | UncheckedIOException
                    | IOException e) {
                spinner.stop();
                System.err.println("ERROR " + Options.rootMessage(e));
                return 1;
            }
            // a native image is compiled ahead of time: nothing to warm
            if (stdin && System.getProperty("org.graalvm.nativeimage.imagecode") == null)
                warmUp(model);
            spinner.stop();
            Terminal terminal = stdin ? Terminal.stderr() : null;
            Transcription transcription =
                    stdin
                            ? pump(model, System.in, terminal, options.theme())
                            : model.transcribe(audio.pcm());
            // the live view already settled the full transcript on this same screen
            if (terminal == null || !Terminal.isTerminal(1))
                System.out.println(transcription.text());
            return 0;
        } catch (IOException e) {
            System.err.println("ERROR reading stdin: " + Options.rootMessage(e));
            return 1;
        } finally {
            Arenas.close(arena);
        }
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
     * of new audio, on the live view only while speech comes in, and everything captured meanwhile
     * is fed before the next one, so a slow decode delays the view but never lets it fall behind.
     */
    private static <S extends RuntimeState> Transcription pump(
            TranscriptionModel<?, ?, S> model,
            InputStream in,
            Terminal terminal,
            TranscriptHud.Theme theme)
            throws IOException {
        TranscriptHud view =
                terminal == null ? null : new TranscriptHud(terminal, Terminal::columns, theme);
        int rate = model.sampleRate();
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
                                readFailure[0] = e;
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
            long partialAt = Long.MIN_VALUE; // when the last partial started
            int logged = 0; // characters of the final text the plain log has printed
            boolean ended = false;
            while (!ended) {
                float[] pcm;
                try {
                    pcm = queue.take();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
                }
                // feed everything captured so far before spending time on a partial
                while (true) {
                    if (pcm == eof) {
                        ended = true;
                        break;
                    }
                    Transcription piece = stream.feed(pcm);
                    fed += pcm.length;
                    fresh += pcm.length;
                    if (!piece.tokens().isEmpty()) {
                        text.append(piece.text());
                        tokens.addAll(piece.tokens());
                        if (view != null) view.show(tokens, tail);
                        else logged = logWords(text, logged, false);
                    }
                    pcm = queue.poll();
                    if (pcm == null) break;
                }
                if (ended) break;
                // new silence cannot change the partial, so it is not decoded again
                boolean speech = view == null || view.heardSince(partialAt);
                if (fresh >= refreshEvery && speech) {
                    fresh = 0;
                    partialAt = System.nanoTime();
                    Transcription partial = stream.partial();
                    tail = partial.tokens();
                    if (view != null) view.show(tokens, tail);
                    else System.err.printf("… %ds %s%n", fed / rate, partial.text().strip());
                }
            }
            if (readFailure[0] != null) throw readFailure[0];
            Transcription last = stream.finish();
            text.append(last.text());
            tokens.addAll(last.tokens());
            Transcription finished = new Transcription(text.toString(), tokens);
            if (view != null) view.finish(finished.words());
            else logWords(text, logged, true);
            return finished;
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
    private static int logWords(StringBuilder text, int logged, boolean all) {
        int end = all ? text.length() : text.lastIndexOf(" "); // a word starts at its space
        if (end <= logged) return logged;
        String words = text.substring(logged, end).strip();
        if (!words.isEmpty()) System.err.println(words);
        return end;
    }
}
