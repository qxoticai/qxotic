package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Unbounded-streaming soak: feeds hours of seeded synthetic program - corpus speech with
 * deterministically injected pathologies (silence blocks, the collapse-prone clip, a language
 * switch, tone sweeps, gain steps) - through one {@link TranscriptionStream}, checking invariants
 * continuously and resource/latency stability over simulated hours.
 *
 * <pre>
 *   StreamSoak --model tdt.gguf --corpus dir-of-flacs --events dir(en.wav,es.wav) \
 *       --hours 6 [--seed 7] [--report out.txt]
 * </pre>
 *
 * <p>Checked while feeding: the committed transcript only grows and always prefixes the partial;
 * token times are ordered and never ahead of the audio; a live mute alarm fires when a minute of
 * speech-energy audio produces no new text. Checked at the end: no 45 s speech span without
 * emissions, recovery within 60 s after every injected event, RSS slope after warmup, and partial
 * latency percentiles per simulated hour. Phase 0 first proves streamed == offline on a 30 min
 * prefix of the same program.
 */
public final class StreamSoak {

    private StreamSoak() {}

    private static final int RATE = 16_000;
    private static final int FEED_CHUNK = RATE / 2; // 0.5 s
    private static final int PARTIAL_EVERY_SECONDS = 10;

    record Event(String type, double startSec, double endSec) {}

    /** The seeded program: which clip or synthetic block plays at which simulated second. */
    static final class Program {
        final List<float[]> blocks = new ArrayList<>();
        final List<Event> events = new ArrayList<>();
        double seconds;

        void add(float[] pcm, String eventType) {
            if (eventType != null)
                events.add(new Event(eventType, seconds, seconds + pcm.length / (double) RATE));
            blocks.add(pcm);
            seconds += pcm.length / (double) RATE;
        }
    }

    public static void main(String[] args) throws Exception {
        Path model = null, corpus = null, eventsDir = null, report = null;
        double hours = 6;
        long seed = 7;
        for (int i = 0; i < args.length; i += 2) {
            switch (args[i]) {
                case "--model" -> model = Path.of(args[i + 1]);
                case "--corpus" -> corpus = Path.of(args[i + 1]);
                case "--events" -> eventsDir = Path.of(args[i + 1]);
                case "--hours" -> hours = Double.parseDouble(args[i + 1]);
                case "--seed" -> seed = Long.parseLong(args[i + 1]);
                case "--report" -> report = Path.of(args[i + 1]);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (model == null || corpus == null || eventsDir == null)
            throw new IllegalArgumentException("--model, --corpus and --events are required");

        List<float[]> clips = loadClips(corpus);
        float[] collapse = AudioCodec.load(eventsDir.resolve("en.wav")).pcm();
        float[] spanish = AudioCodec.load(eventsDir.resolve("es.wav")).pcm();
        StringBuilder out = new StringBuilder();

        try (Arena arena = Arena.ofShared()) {
            TranscriptionModel<?, ?, ?> transcriber = Models.loadTranscription(model, arena);

            // ---- phase 0: streamed == offline on a 30 min prefix of the same program ----
            Program prefix = program(clips, collapse, spanish, seed, 0.5 * 3600);
            float[] prefixPcm = concat(prefix);
            String offline = transcriber.transcribe(prefixPcm).text();
            String streamed = streamWhole(transcriber, prefixPcm);
            line(
                    out,
                    "phase0 streamed==offline over %.0f min: %s",
                    prefix.seconds / 60,
                    offline.equals(streamed) ? "EQUAL" : "MISMATCH");

            // ---- the soak ----
            soak(transcriber, program(clips, collapse, spanish, seed, hours * 3600), out);
        }
        System.out.print(out);
        if (report != null) Files.writeString(report, out.toString());
    }

    private static void soak(
            TranscriptionModel<?, ?, ?> transcriber, Program program, StringBuilder out)
            throws Exception {
        long wallStart = System.nanoTime();
        List<double[]> rss = new ArrayList<>(); // (simMinutes, rssMb)
        Map<Integer, List<Long>> partialNanosByHour = new HashMap<>();
        List<String> violations = new ArrayList<>();
        boolean[] speechBySecond = new boolean[(int) program.seconds + 2];
        double lastGrowthSec = 0;
        int lastPartialLength = 0;
        List<Transcription.Token> finalTokens = null;
        String previousCommitted = "";
        double fedSec = 0;
        long nextPartialAt = PARTIAL_EVERY_SECONDS * (long) RATE;
        long nextRssAt = 0;
        long fedSamples = 0;

        try (AutoCloseable state = (AutoCloseable) newState(transcriber);
                TranscriptionStream stream = stream(transcriber, state)) {
            for (float[] block : program.blocks) {
                for (int from = 0; from < block.length; from += FEED_CHUNK) {
                    int n = Math.min(FEED_CHUNK, block.length - from);
                    markSpeech(speechBySecond, block, from, n, fedSamples);
                    stream.feed(block, from, n);
                    fedSamples += n;
                    fedSec = fedSamples / (double) RATE;

                    if (fedSamples >= nextPartialAt) {
                        nextPartialAt += PARTIAL_EVERY_SECONDS * (long) RATE;
                        long t0 = System.nanoTime();
                        Transcription partial = stream.partial();
                        long nanos = System.nanoTime() - t0;
                        partialNanosByHour
                                .computeIfAbsent((int) (fedSec / 3600), h -> new ArrayList<>())
                                .add(nanos);
                        String committed = stream.committed().text();
                        if (!committed.startsWith(previousCommitted))
                            violations.add("committed shrank at %.0fs".formatted(fedSec));
                        if (!partial.text().startsWith(committed))
                            violations.add("committed not a prefix at %.0fs".formatted(fedSec));
                        previousCommitted = committed;
                        List<Transcription.Token> tokens = partial.tokens();
                        if (!tokens.isEmpty()) {
                            Transcription.Token last = tokens.get(tokens.size() - 1);
                            if (last.end() > fedSec + 2)
                                violations.add(
                                        "token ahead of audio at %.0fs (end %.1f)"
                                                .formatted(fedSec, last.end()));
                        }
                        double ordered = -1;
                        for (Transcription.Token token : tokens) {
                            if (token.start() < ordered) {
                                violations.add("token order regressed at %.0fs".formatted(fedSec));
                                break;
                            }
                            ordered = token.start();
                        }
                        if (partial.text().length() > lastPartialLength) {
                            lastPartialLength = partial.text().length();
                            lastGrowthSec = fedSec;
                        } else if (fedSec - lastGrowthSec > 60
                                && speechFraction(speechBySecond, fedSec - 60, fedSec) > 0.5) {
                            violations.add(
                                    "mute alarm: no growth %.0f..%.0fs with speech"
                                            .formatted(lastGrowthSec, fedSec));
                            lastGrowthSec = fedSec; // one report per incident
                        }
                    }
                    if (fedSamples >= nextRssAt) {
                        nextRssAt += 300L * RATE; // every 5 simulated minutes
                        rss.add(new double[] {fedSec / 60, rssMb()});
                    }
                }
            }
            Transcription finished = stream.finish();
            finalTokens = finished.tokens();
        }

        double wallMinutes = (System.nanoTime() - wallStart) / 60e9;
        line(
                out,
                "soak: %.2f h simulated in %.1f min wall (%.1fx realtime)",
                program.seconds / 3600,
                wallMinutes,
                program.seconds / 60 / wallMinutes);
        line(out, "final transcript: %d tokens", finalTokens.size());

        // end-of-run sweep: speech spans without emissions, event recovery
        boolean[] tokenBySecond = new boolean[(int) program.seconds + 2];
        for (Transcription.Token token : finalTokens) {
            int second = (int) token.start();
            if (second < tokenBySecond.length) tokenBySecond[second] = true;
        }
        int holes = 0;
        for (int from = 0; from + 45 < tokenBySecond.length; from++) {
            if (speechFraction(speechBySecond, from, from + 45) > 0.8
                    && none(tokenBySecond, from, from + 45)) {
                violations.add("silent hole %ds..%ds".formatted(from, from + 45));
                holes++;
                from += 45;
            }
        }
        line(out, "speech holes >=45s: %d", holes);
        for (Event event : program.events) {
            double firstAfter = Double.MAX_VALUE;
            for (Transcription.Token token : finalTokens)
                if (token.start() >= event.endSec()) {
                    firstAfter = token.start();
                    break;
                }
            double recovery = firstAfter - event.endSec();
            if (recovery > 60 && firstAfter != Double.MAX_VALUE)
                violations.add(
                        "slow recovery %.0fs after %s@%.0fs"
                                .formatted(recovery, event.type(), event.startSec()));
        }
        line(
                out,
                "events injected: %d (%s)",
                program.events.size(),
                program.events.stream()
                        .map(Event::type)
                        .distinct()
                        .sorted()
                        .reduce((a, b) -> a + "," + b)
                        .orElse(""));

        // resource + latency stability
        if (rss.size() > 6) {
            double slope = slopePerHour(rss.subList(rss.size() / 3, rss.size()));
            line(
                    out,
                    "rss: start %.0f MB, end %.0f MB, post-warmup slope %+.1f MB/simulated-hour",
                    rss.get(0)[1],
                    rss.get(rss.size() - 1)[1],
                    slope);
        }
        for (int hour : partialNanosByHour.keySet().stream().sorted().toList()) {
            List<Long> nanos = partialNanosByHour.get(hour).stream().sorted().toList();
            line(
                    out,
                    "partial latency hour %d: p50 %.2fs  p99 %.2fs  (n=%d)",
                    hour,
                    nanos.get(nanos.size() / 2) / 1e9,
                    nanos.get((int) (nanos.size() * 0.99)) / 1e9,
                    nanos.size());
        }
        line(out, "violations: %d", violations.size());
        for (String violation : violations) line(out, "  ! %s", violation);
    }

    // ---- seeded program construction ----

    static Program program(
            List<float[]> clips, float[] collapse, float[] spanish, long seed, double seconds) {
        Random random = new Random(seed);
        Program program = new Program();
        int slot = 0;
        while (program.seconds < seconds) {
            slot++;
            if (slot % 13 == 0) program.add(new float[(60 + random.nextInt(60)) * RATE], "silence");
            else if (slot % 17 == 0) program.add(collapse, "collapse-clip");
            else if (slot % 19 == 0) program.add(spanish, "language-switch");
            else if (slot % 23 == 0) program.add(toneSweep(30), "tone-sweep");
            else {
                float[] clip = clips.get(random.nextInt(clips.size()));
                if (slot % 11 == 0) { // gain step: same speech at 0.4x level
                    float[] soft = clip.clone();
                    for (int i = 0; i < soft.length; i++) soft[i] *= 0.4f;
                    program.add(soft, "gain-step");
                } else {
                    program.add(clip, null);
                }
            }
        }
        return program;
    }

    private static float[] toneSweep(int seconds) {
        float[] tone = new float[seconds * RATE];
        double phase = 0;
        for (int i = 0; i < tone.length; i++) {
            double hz = 200 + 1800.0 * i / tone.length;
            phase += 2 * Math.PI * hz / RATE;
            tone[i] = (float) (0.1 * Math.sin(phase));
        }
        return tone;
    }

    static List<float[]> loadClips(Path corpus) throws IOException {
        try (Stream<Path> files = Files.list(corpus)) {
            List<float[]> clips = new ArrayList<>();
            for (Path flac : files.filter(f -> f.toString().endsWith(".flac")).sorted().toList())
                clips.add(AudioCodec.load(flac).pcm());
            if (clips.isEmpty()) throw new IllegalArgumentException("no .flac clips in " + corpus);
            return clips;
        }
    }

    private static float[] concat(Program program) {
        int total = program.blocks.stream().mapToInt(b -> b.length).sum();
        float[] all = new float[total];
        int at = 0;
        for (float[] block : program.blocks) {
            System.arraycopy(block, 0, all, at, block.length);
            at += block.length;
        }
        return all;
    }

    // ---- helpers ----

    @SuppressWarnings("unchecked")
    private static <S extends RuntimeState> S newState(TranscriptionModel<?, ?, ?> model) {
        return ((TranscriptionModel<?, ?, S>) model).newState();
    }

    @SuppressWarnings("unchecked")
    private static <S extends RuntimeState> TranscriptionStream stream(
            TranscriptionModel<?, ?, ?> model, Object state) {
        return ((TranscriptionModel<?, ?, S>) model).stream((S) state);
    }

    private static String streamWhole(TranscriptionModel<?, ?, ?> transcriber, float[] pcm)
            throws Exception {
        try (AutoCloseable state = (AutoCloseable) newState(transcriber);
                TranscriptionStream stream = stream(transcriber, state)) {
            for (int from = 0; from < pcm.length; from += FEED_CHUNK)
                stream.feed(pcm, from, Math.min(FEED_CHUNK, pcm.length - from));
            return stream.finish().text();
        }
    }

    private static void markSpeech(
            boolean[] bySecond, float[] block, int from, int n, long fedSamples) {
        double sum = 0;
        for (int i = from; i < from + n; i++) sum += block[i] * block[i];
        double rms = Math.sqrt(sum / n);
        if (rms > 0.02) {
            int second = (int) (fedSamples / RATE);
            if (second < bySecond.length) bySecond[second] = true;
        }
    }

    private static double speechFraction(boolean[] bySecond, double fromSec, double toSec) {
        int from = Math.max(0, (int) fromSec), to = Math.min(bySecond.length, (int) toSec);
        if (to <= from) return 0;
        int speech = 0;
        for (int s = from; s < to; s++) if (bySecond[s]) speech++;
        return speech / (double) (to - from);
    }

    private static boolean none(boolean[] values, int from, int to) {
        for (int i = from; i < Math.min(to, values.length); i++) if (values[i]) return false;
        return true;
    }

    private static double rssMb() {
        try {
            Process ps =
                    new ProcessBuilder(
                                    "ps",
                                    "-o",
                                    "rss=",
                                    "-p",
                                    Long.toString(ProcessHandle.current().pid()))
                            .start();
            String rss = new String(ps.getInputStream().readAllBytes()).strip();
            return Long.parseLong(rss) / 1024.0;
        } catch (Exception e) {
            return -1;
        }
    }

    private static double slopePerHour(List<double[]> points) {
        double n = points.size(), sx = 0, sy = 0, sxy = 0, sxx = 0;
        for (double[] point : points) {
            sx += point[0];
            sy += point[1];
            sxy += point[0] * point[1];
            sxx += point[0] * point[0];
        }
        double perMinute = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        return perMinute * 60;
    }

    private static void line(StringBuilder out, String format, Object... args) {
        String line = String.format(Locale.ROOT, format, args);
        System.out.println(line);
        out.append(line).append('\n');
    }
}
