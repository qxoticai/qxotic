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
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * {@code --transcribe <audio|->}: transcript of an audio file, or {@code -} to stream raw 16 kHz
 * mono s16le PCM from stdin with live partials on stderr - one status line redrawn in place when
 * stderr is a terminal, one line per partial when redirected. The final transcript alone goes to
 * stdout, so it pipes. Pipe a microphone in with e.g. {@code ffmpeg -nostats -loglevel error -f
 * avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - | jinfer -m parakeet.gguf --transcribe -}
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
            spinner.stop();
            boolean hud = stdin && terminal(2);
            Transcription transcription =
                    stdin ? pump(model, System.in, hud) : model.transcribe(audio.pcm());
            // the live view already settled the full transcript on this same screen
            if (!(hud && terminal(1))) System.out.println(transcription.text());
            return 0;
        } catch (IOException e) {
            System.err.println("ERROR reading stdin: " + Options.rootMessage(e));
            return 1;
        } finally {
            Arenas.close(arena);
        }
    }

    /** Streams stdin PCM through the model, surfacing a partial every two seconds of audio. */
    private static <S extends RuntimeState> Transcription pump(
            TranscriptionModel<?, ?, S> model, InputStream in, boolean hud) throws IOException {
        TranscriptHud view = hud ? new TranscriptHud(System.err) : null;
        int partialEvery = 2 * model.sampleRate();
        // Decoding pauses for tail re-decodes and window commits, but a live source cannot: if
        // this thread stops reading, the pipe backs up and the capture side drops microphone
        // audio - which reaches the model as spliced garbage. A reader thread keeps stdin
        // drained; the queue absorbs decode bursts. Bounded, so a decode that cannot keep up at
        // all backpressures like any pipe instead of buffering without limit.
        BlockingQueue<float[]> queue = new ArrayBlockingQueue<>(4096); // ~13 min of 0.2 s chunks
        float[] eof = new float[0];
        IOException[] readFailure = new IOException[1];
        Thread reader =
                new Thread(
                        () -> {
                            byte[] bytes = new byte[6400]; // 0.2 s of s16le at 16 kHz
                            try {
                                int read;
                                while ((read = in.readNBytes(bytes, 0, bytes.length)) > 0) {
                                    int samples = read / 2;
                                    float[] pcm = new float[samples];
                                    for (int i = 0; i < samples; i++) {
                                        int lo = bytes[2 * i] & 0xFF, hi = bytes[2 * i + 1];
                                        pcm[i] = ((short) ((hi << 8) | lo)) / 32768f;
                                    }
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
            long fed = 0;
            long sincePartial = 0;
            boolean ended = false;
            while (!ended) {
                float[] pcm;
                try {
                    pcm = queue.take();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
                }
                // drain everything already captured before spending time on a partial
                while (true) {
                    if (pcm == eof) {
                        ended = true;
                        break;
                    }
                    stream.feed(pcm);
                    fed += pcm.length;
                    sincePartial += pcm.length;
                    pcm = queue.poll();
                    if (pcm == null) break;
                }
                if (!ended && sincePartial >= partialEvery) {
                    sincePartial = 0;
                    long seconds = fed / model.sampleRate();
                    String text = stream.partial().text();
                    if (view != null) {
                        view.render(stream.committed().text(), text, seconds);
                    } else {
                        // the elapsed stamp keeps the heartbeat visible even over silence
                        System.err.printf("… %ds %s%n", seconds, text);
                    }
                }
            }
            if (readFailure[0] != null) throw readFailure[0];
            Transcription finished = stream.finish();
            if (view != null) view.finish(finished.text());
            return finished;
        }
    }

    /**
     * Whether the file descriptor is an interactive terminal. {@link java.io.Console#isTerminal}
     * answers for stdin/stdout jointly, and streaming mode always pipes stdin, so ask libc
     * directly. False on any failure - a missing native-image descriptor degrades to plain line
     * output, never breaks.
     */
    private static boolean terminal(int fd) {
        try {
            Linker linker = Linker.nativeLinker();
            var isatty =
                    linker.downcallHandle(
                            linker.defaultLookup().find("isatty").orElseThrow(),
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            return (int) isatty.invokeExact(fd) == 1;
        } catch (Throwable unsupported) {
            return false;
        }
    }
}
