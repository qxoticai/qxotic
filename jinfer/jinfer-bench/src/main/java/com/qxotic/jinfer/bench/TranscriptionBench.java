package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Speech-recognition quality and speed:
 *
 * <pre>
 *   WER  over a LibriSpeech-layout corpus (*.flac beside *.trans.txt):
 *        TranscriptionBench --model tdt.gguf --librispeech LibriSpeech/test-clean [--limit N]
 *                           [--dump out.tsv]
 *   RTF  over one audio file:
 *        TranscriptionBench --model tdt.gguf --audio jfk.wav [--reps N]
 * </pre>
 *
 * <p>{@code --gate <percent>} turns the WER run into a pass/fail check: exit 1 when the corpus WER
 * exceeds the bound. Decoding is deterministic, so a subset's WER is a constant per model and
 * quant; gate with a margin over the measured value and a windowing or normalization regression
 * trips it while quantization noise cannot.
 *
 * <p>{@code --dump <file>} writes what was heard, one utterance per line: {@code id, audio seconds,
 * decode seconds, hypothesis, reference}, tab separated, so another scorer can compare engines on
 * equal terms.
 *
 * <p>WER uses the usual normalization (lowercase, keep {@code [a-z0-9']}, collapse spaces) and
 * word-level edit distance. RTF is seconds of audio transcribed per wall second, model load
 * excluded, first run discarded as warmup.
 */
public final class TranscriptionBench {

    private TranscriptionBench() {}

    public static void main(String[] args) throws Exception {
        Path model = null, corpus = null, audio = null, dump = null;
        int limit = Integer.MAX_VALUE, reps = 3;
        double gate = -1;
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length)
                throw new IllegalArgumentException("missing value: " + args[i]);
            switch (args[i]) {
                case "--model", "-m" -> model = Path.of(args[i + 1]);
                case "--librispeech" -> corpus = Path.of(args[i + 1]);
                case "--audio" -> audio = Path.of(args[i + 1]);
                case "--limit" -> limit = Integer.parseInt(args[i + 1]);
                case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                case "--gate" -> gate = Double.parseDouble(args[i + 1]);
                case "--dump" -> dump = Path.of(args[i + 1]);
                default -> throw new IllegalArgumentException("unknown option: " + args[i]);
            }
        }
        if (model == null || (corpus == null) == (audio == null) || (gate >= 0 && corpus == null))
            throw new IllegalArgumentException(
                    "usage: --model <gguf> (--librispeech <dir> [--limit N] [--gate maxWer%]"
                            + " [--dump out.tsv] |"
                            + " --audio <file> [--reps N])");

        try (Arena arena = Arena.ofShared()) {
            long loadStart = System.nanoTime();
            TranscriptionModel<?, ?, ?> transcriber = Models.loadTranscription(model, arena);
            System.out.printf(
                    "model %s loaded in %.1f s%n",
                    model.getFileName(), (System.nanoTime() - loadStart) / 1e9);
            if (corpus != null) {
                double wer = wer(transcriber, corpus, limit, dump);
                if (gate >= 0) {
                    boolean passed = wer <= gate;
                    System.out.printf(
                            "%ngate %.3f%%: %s (measured %.3f%%)%n",
                            gate, passed ? "PASS" : "FAIL", wer);
                    if (!passed) System.exit(1);
                }
            } else {
                rtf(transcriber, audio, reps);
            }
        }
    }

    private record Utterance(String id, Path flac, String reference) {}

    private record Scored(
            Utterance utterance,
            int errors,
            int words,
            String hypothesis,
            double audioSeconds,
            double decodeSeconds) {}

    private static <S extends RuntimeState> double wer(
            TranscriptionModel<?, ?, S> transcriber, Path corpus, int limit, Path dump)
            throws IOException {
        List<Utterance> utterances = corpus(corpus, limit);
        System.out.printf("%d utterances from %s%n", utterances.size(), corpus);
        long errors = 0, words = 0, samples = 0, nanos = 0;
        List<Scored> scored = new ArrayList<>(utterances.size());
        // one state for the corpus, as a server holds one per pipeline: the timed region is
        // decoding, not allocating a fresh workspace per utterance
        try (S state = transcriber.newState()) {
            for (int i = 0; i < utterances.size(); i++) {
                Utterance utterance = utterances.get(i);
                Media.Audio decoded = AudioCodec.load(utterance.flac());
                long start = System.nanoTime();
                Transcription transcription = transcriber.transcribe(state, decoded.pcm());
                long spent = System.nanoTime() - start;
                nanos += spent;
                samples += decoded.pcm().length;
                String[] reference = normalize(utterance.reference());
                String[] hypothesis = normalize(transcription.text());
                int distance = editDistance(reference, hypothesis);
                errors += distance;
                words += reference.length;
                scored.add(
                        new Scored(
                                utterance,
                                distance,
                                reference.length,
                                transcription.text(),
                                decoded.pcm().length / (double) transcriber.sampleRate(),
                                spent / 1e9));
                if ((i + 1) % 100 == 0 || i + 1 == utterances.size())
                    System.out.printf(
                            "  %5d/%d  WER %.3f%%  RTFx %.1f%n",
                            i + 1,
                            utterances.size(),
                            100.0 * errors / Math.max(1, words),
                            (samples / 16_000.0) / (nanos / 1e9));
            }
        }
        System.out.printf(
                "%nWER %.3f%%  (%d errors / %d words, %d utterances)%n",
                100.0 * errors / Math.max(1, words), errors, words, utterances.size());
        System.out.printf(
                "speed: %.1f s of audio per second (%.1f min in %.1f min)%n",
                (samples / 16_000.0) / (nanos / 1e9), samples / 16_000.0 / 60, nanos / 1e9 / 60);
        System.out.println("\nworst utterances:");
        scored.stream()
                .sorted(
                        Comparator.comparingDouble(
                                        (Scored s) -> (double) s.errors() / Math.max(1, s.words()))
                                .reversed())
                .limit(5)
                .forEach(
                        s ->
                                System.out.printf(
                                        "  %s  %d/%d%n    ref: %s%n    hyp: %s%n",
                                        s.utterance().id(),
                                        s.errors(),
                                        s.words(),
                                        s.utterance().reference(),
                                        s.hypothesis()));
        if (dump != null) {
            List<String> lines = new ArrayList<>(scored.size());
            for (Scored s : scored)
                lines.add(
                        "%s\t%.3f\t%.3f\t%s\t%s"
                                .formatted(
                                        s.utterance().id(),
                                        s.audioSeconds(),
                                        s.decodeSeconds(),
                                        s.hypothesis().strip(),
                                        s.utterance().reference()));
            Files.write(dump, lines);
            System.out.println("dumped " + lines.size() + " utterances to " + dump);
        }
        return 100.0 * errors / Math.max(1, words);
    }

    private static <S extends RuntimeState> void rtf(
            TranscriptionModel<?, ?, S> transcriber, Path audio, int reps) throws IOException {
        Media.Audio decoded = AudioCodec.load(audio);
        double seconds = decoded.pcm().length / (double) transcriber.sampleRate();
        try (S state = transcriber.newState()) {
            transcriber.transcribe(state, decoded.pcm()); // warmup
            for (int i = 0; i < reps; i++) {
                long start = System.nanoTime();
                Transcription transcription = transcriber.transcribe(state, decoded.pcm());
                double wall = (System.nanoTime() - start) / 1e9;
                System.out.printf(
                        "run %d: %.2f s audio in %.2f s  (RTFx %.1f)%n",
                        i + 1, seconds, wall, seconds / wall);
                if (i == 0) System.out.println("  " + transcription.text());
            }
        }
    }

    /** LibriSpeech layout: {@code chapter/*.trans.txt} lines of {@code <id> <TRANSCRIPT>}. */
    private static List<Utterance> corpus(Path corpus, int limit) throws IOException {
        List<Utterance> utterances = new ArrayList<>();
        try (Stream<Path> transcripts = Files.walk(corpus)) {
            List<Path> files =
                    transcripts
                            .filter(p -> p.getFileName().toString().endsWith(".trans.txt"))
                            .sorted()
                            .toList();
            for (Path transcript : files) {
                for (String line : Files.readAllLines(transcript)) {
                    int space = line.indexOf(' ');
                    if (space < 0) continue;
                    String id = line.substring(0, space);
                    Path flac = transcript.getParent().resolve(id + ".flac");
                    if (Files.isRegularFile(flac))
                        utterances.add(new Utterance(id, flac, line.substring(space + 1)));
                    if (utterances.size() >= limit) return utterances;
                }
            }
        }
        return utterances;
    }

    static String[] normalize(String text) {
        String cleaned =
                text.toLowerCase(Locale.ROOT)
                        .replaceAll("[^a-z0-9' ]", " ")
                        .replaceAll("\\s+", " ")
                        .strip();
        return cleaned.isEmpty() ? new String[0] : cleaned.split(" ");
    }

    /** Word-level Levenshtein distance, two-row DP. */
    static int editDistance(String[] reference, String[] hypothesis) {
        int[] previous = new int[hypothesis.length + 1];
        int[] current = new int[hypothesis.length + 1];
        for (int j = 0; j <= hypothesis.length; j++) previous[j] = j;
        for (int i = 1; i <= reference.length; i++) {
            current[0] = i;
            for (int j = 1; j <= hypothesis.length; j++) {
                int substitute =
                        previous[j - 1] + (reference[i - 1].equals(hypothesis[j - 1]) ? 0 : 1);
                current[j] = Math.min(substitute, Math.min(previous[j] + 1, current[j - 1] + 1));
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[hypothesis.length];
    }
}
