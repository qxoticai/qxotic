// Inflect2 TTS convenience layer — text splitting, phonemization, and Media.Audio output.
//   InflectTTS tts = InflectTTS.load(Path.of("model.gguf"));
//   Media.Audio audio = tts.synthesize("Hello world.", 1.0, 0.667, 42);
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.models.inflect2.frontend.Phonemizer;
import com.qxotic.jinfer.models.inflect2.frontend.TextNormalizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;

public final class InflectTTS {
    private final Inflect2 model;
    private final Inflect2.State state;
    private final Phonemizer phonemizer;
    private volatile Map<String, String> wordOverrides = Map.of();

    private InflectTTS(Inflect2 model) {
        this.model = model;
        this.state = model.newState();
        this.phonemizer = Phonemizer.tryCreate();
    }

    // ── word overrides ──────────────────────────────────────────────────

    /**
     * Set pronunciation overrides — terms expanded to readable English before phonemization.
     * Applied on the next {@link #synthesize} call. Built-in overrides (e.g. {@code "PyTorch"→"pie
     * torch"}) are always present; user entries here take priority.
     */
    public void setWordOverrides(Map<String, String> overrides) {
        this.wordOverrides = Map.copyOf(overrides);
    }

    public Inflect2.Configuration config() {
        return model.config();
    }

    public Inflect2.Weights weights() {
        return model.weights();
    }

    public Inflect2 model() {
        return model;
    }

    public int sampleRate() {
        return model.sampleRate();
    }

    public static InflectTTS load(Path ggufPath) throws IOException {
        return new InflectTTS(Inflect2.load(ggufPath));
    }

    /** Load from embedded classpath resource (native image bundles .gguf files). */
    public static InflectTTS loadResource(String resourcePath) throws IOException {
        return new InflectTTS(Inflect2.loadResource(resourcePath));
    }

    /**
     * Load from a ZIP overlay appended to the running executable. {@code entryName} is a path
     * within the ZIP, e.g. {@code "default.gguf"} or {@code "nano_q8.gguf"}. Symlinks are resolved
     * transparently.
     */
    public static InflectTTS loadSelfArchive(String entryName) throws IOException {
        return new InflectTTS(Inflect2.loadSelfArchive(entryName));
    }

    // ── synthesis ─────────────────────────────────────────────────────

    /** Synthesize text → Media.Audio (24kHz mono PCM in [-1,1]). */
    public Media.Audio synthesize(String text, double speed, double variation, long seed)
            throws Exception {
        List<float[]> pieces = new ArrayList<>();
        try (AudioStream as = stream(text, speed, variation, seed)) {
            for (float[] chunk : as) pieces.add(chunk);
        }
        return new Media.Audio(joinAndClip(pieces), model.sampleRate(), 1);
    }

    /**
     * Lazy pull-based stream of synthesized audio chunks (one per sentence). Synthesis happens
     * on-demand in the calling thread as the stream is consumed.
     *
     * <pre>{@code
     * try (var stream = tts.stream("Hello.", 1.0, 0.667, 42)) {
     *     stream.forEach(chunk -> writeToPipe(chunk));
     * }
     * }</pre>
     */
    public AudioStream stream(String text, double speed, double variation, long seed)
            throws Exception {
        var overrides = this.wordOverrides;
        List<String> chunks = splitText(text);
        Iterator<float[]> it =
                new Iterator<>() {
                    int i = 0;

                    @Override
                    public boolean hasNext() {
                        return i < chunks.size();
                    }

                    @Override
                    public float[] next() {
                        try {
                            String chunk = TextNormalizer.normalize(chunks.get(i), overrides);
                            int[] tokens =
                                    phonemizer != null
                                            ? phonemizer.phonemize(chunk)
                                            : Symbols.toTokens(chunk);
                            float[] wav =
                                    edgeFade(
                                            model.synthesize(
                                                            state,
                                                            tokens,
                                                            (float) (1.0 / speed),
                                                            (float) variation,
                                                            seed + i)
                                                    .pcm());
                            i++;
                            return wav;
                        } catch (Exception e) {
                            throw new UncheckedIOException(new IOException("synthesis failed", e));
                        }
                    }
                };
        return AudioStream.sync(it);
    }

    /**
     * Async stream — synthesis runs on a background thread with 1-chunk pipeline overlap. Chunk N+1
     * is synthesized while the caller processes chunk N.
     */
    public AudioStream streamAsync(String text, double speed, double variation, long seed)
            throws Exception {
        var overrides = this.wordOverrides;
        List<String> chunks = splitText(text);
        ArrayBlockingQueue<float[]> q = new ArrayBlockingQueue<>(2);
        Inflect2.State synthState = model.newState(); // dedicated state for background thread
        Thread synth =
                new Thread(
                        () -> {
                            try {
                                for (int i = 0; i < chunks.size(); i++) {
                                    String chunk =
                                            TextNormalizer.normalize(chunks.get(i), overrides);
                                    int[] tokens =
                                            phonemizer != null
                                                    ? phonemizer.phonemize(chunk)
                                                    : Symbols.toTokens(chunk);
                                    float[] wav =
                                            edgeFade(
                                                    model.synthesize(
                                                                    synthState,
                                                                    tokens,
                                                                    (float) (1.0 / speed),
                                                                    (float) variation,
                                                                    seed + i)
                                                            .pcm());
                                    q.put(wav);
                                }
                            } catch (Exception e) {
                                q.offer(new float[0]);
                            }
                        },
                        "inflect-synth");
        synth.setDaemon(true);
        synth.start();
        return AudioStream.async(q, synth, chunks.size());
    }

    public void save(String text, Path out, double speed, double variation, long seed)
            throws Exception {
        AudioIO.writeWav(synthesize(text, speed, variation, seed).pcm(), model.sampleRate(), out);
    }

    // ── playback ────────────────────────────────────────────────────

    /**
     * Play synthesized audio in real time — starts a subprocess player and streams chunks as
     * they're produced. Synthesis of chunk N+1 overlaps with playback of chunk N.
     */
    public void play(String text, double speed, double variation, long seed) throws Exception {
        long t0 = System.currentTimeMillis();
        String[] cmd = playCommand(model.sampleRate());
        Process proc =
                new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        int totalSamples = 0;
        boolean first = true;
        try (var out = proc.getOutputStream();
                AudioStream as = streamAsync(text, speed, variation, seed)) {
            out.write(AudioIO.toS16LE(silence(0.03)));
            for (float[] chunk : as) {
                if (first) {
                    first = false;
                    long ttf = System.currentTimeMillis() - t0;
                    System.out.printf(
                            "ttf: %d ms, first chunk: %d samples (%.2f s)%n",
                            ttf, chunk.length, chunk.length / (float) model.sampleRate());
                }
                totalSamples += chunk.length;
                out.write(AudioIO.toS16LE(chunk));
                out.flush();
            }
        }
        long dt = System.currentTimeMillis() - t0;
        double rtf = totalSamples / (double) model.sampleRate() / (dt / 1000.0);
        System.out.printf(
                "synthesis: %d ms, %d samples (%.2f s), %.2f× realtime%n",
                dt, totalSamples, totalSamples / (float) model.sampleRate(), rtf);
        try {
            proc.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            proc.destroy();
        }
    }

    private static boolean commandExists(String cmd) {
        try {
            return new ProcessBuilder("which", cmd)
                            .redirectError(ProcessBuilder.Redirect.DISCARD)
                            .start()
                            .waitFor()
                    == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    private static String[] playCommand(int sampleRate) throws IOException {
        boolean linux = System.getProperty("os.name", "").toLowerCase().contains("linux");
        // aplay on Linux: raw S16_LE, no header, reads from stdin
        if (linux && commandExists("aplay")) {
            return new String[] {
                "aplay", "-f", "S16_LE", "-r", String.valueOf(sampleRate), "-c", "1", "-"
            };
        }
        // ffplay: works everywhere, reads raw sample pipe
        if (commandExists("ffplay")) {
            return new String[] {
                "ffplay",
                "-f",
                "s16le",
                "-ar",
                String.valueOf(sampleRate),
                "-ac",
                "1",
                "-nodisp",
                "-autoexit",
                "-"
            };
        }
        throw new IOException("no audio player found — install aplay or ffplay");
    }

    private static byte[] pcmToS16LE(float[] pcm) {
        byte[] buf = new byte[pcm.length * 2];
        for (int i = 0; i < pcm.length; i++) {
            int s = (int) (pcm[i] * 32767);
            if (s > 32767) s = 32767;
            else if (s < -32768) s = -32768;
            buf[i * 2] = (byte) s;
            buf[i * 2 + 1] = (byte) (s >> 8);
        }
        return buf;
    }

    // ── text splitting ────────────────────────────────────────────────────

    private float[] silence(double sec) {
        return new float[(int) (model.sampleRate() * sec)];
    }

    private float[] edgeFade(float[] wav) {
        int frames = Math.min(model.sampleRate() * 5 / 1000, wav.length / 2);
        if (frames <= 0) return wav;
        float[] out = wav.clone();
        for (int i = 0; i < frames; i++) {
            float r = (float) i / frames;
            out[i] *= r;
            out[out.length - 1 - i] *= r;
        }
        return out;
    }

    private static List<String> splitText(String text) {
        String norm = text.replaceAll("\\s+", " ").trim();
        String[] sents = norm.split("(?<=[.!?;:])\\s+");
        List<String> chunks = new ArrayList<>();
        for (String sent : sents) {
            String s = sent.trim();
            if (s.isEmpty()) continue;
            while (s.length() > 280) {
                int split = s.length();
                for (char c : new char[] {',', ';', ':'}) {
                    int p = s.lastIndexOf(c, 281);
                    if (p >= 140) {
                        split = p + 1;
                        break;
                    }
                }
                if (split >= 280) {
                    int sp = s.lastIndexOf(' ', 281);
                    split = sp >= 140 ? sp : 280;
                }
                chunks.add(s.substring(0, split).trim());
                s = s.substring(split).trim();
            }
            if (!s.isEmpty()) chunks.add(s);
        }
        return chunks.isEmpty() ? List.of(norm) : chunks;
    }

    private static double boundaryPause(String chunk) {
        if (chunk.isEmpty()) return 0.08;
        return switch (chunk.charAt(chunk.length() - 1)) {
            case '?' -> 0.28;
            case '!' -> 0.24;
            case '.' -> 0.22;
            case ';' -> 0.16;
            case ':' -> 0.13;
            case ',' -> 0.09;
            default -> 0.08;
        };
    }

    private static float[] joinAndClip(List<float[]> pieces) {
        int total = 0;
        for (float[] p : pieces) total += p.length;
        float[] out = new float[total];
        int off = 0;
        for (float[] p : pieces) {
            System.arraycopy(p, 0, out, off, p.length);
            off += p.length;
        }
        for (int i = 0; i < out.length; i++) out[i] = Math.clamp(out[i], -1f, 1f);
        return out;
    }

    // ── CLI ──────────────────────────────────────────────────────────────

    private static void showUsage() {
        System.err.println("usage: inflect [model.gguf] [flags]");
        System.err.println("  --model <path>    model file or z:// entry (default: default.gguf)");
        System.err.println("  --text <string>   text to speak (default: Hello world.)");
        System.err.println("  --output <path>   output WAV file (default: output.wav)");
        System.err.println("  --speed <0.5-2.0> playback speed (default: 1.0)");
        System.err.println("  --variation <0-1> latent noise scale (default: 0.667)");
        System.err.println("  --seed <int>      random seed (default: 7)");
        System.err.println("  --override <k> <v> pronunciation override");
        System.err.println("  --list            list models in self-archive + their config");
        System.err.println("  --play            play audio (pipe to aplay/ffplay)");
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            showUsage();
            return;
        }

        String model = null, text = "Hello world.", out = "output.wav";
        double speed = 1.0, variation = 0.667;
        long seed = 7;
        boolean list = false, play = false;
        var overrides = new java.util.LinkedHashMap<String, String>();

        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if ("--model".equals(a) && i + 1 < args.length) model = args[++i];
            else if ("--text".equals(a) && i + 1 < args.length) text = args[++i];
            else if ("--output".equals(a) && i + 1 < args.length) out = args[++i];
            else if ("--speed".equals(a) && i + 1 < args.length)
                speed = Double.parseDouble(args[++i]);
            else if ("--variation".equals(a) && i + 1 < args.length)
                variation = Double.parseDouble(args[++i]);
            else if ("--seed".equals(a) && i + 1 < args.length) seed = Long.parseLong(args[++i]);
            else if ("--list".equals(a)) list = true;
            else if ("--play".equals(a)) play = true;
            else if ("--override".equals(a) && i + 2 < args.length)
                overrides.put(args[++i], args[++i]);
            else if (!a.startsWith("--") && model == null) model = a; // positional: model path
            else {
                System.err.println("unknown flag: " + a);
                return;
            }
        }

        // ── model resolution ────────────────────────────────────────
        InflectTTS tts = null;
        boolean explicitModel = model != null;
        if (!explicitModel) {
            // Try self-archive default; fall back to usage if not found
            try {
                tts = InflectTTS.loadSelfArchive("default.gguf");
            } catch (IOException ignored) {
            }
            if (tts == null && !list) {
                showUsage();
                return;
            }
        } else if (model.startsWith("z://")) {
            String entryName = model.substring(4);
            if (!entryName.isEmpty()) {
                try {
                    tts = InflectTTS.loadSelfArchive(entryName);
                } catch (IOException ignored) {
                }
            }
        } else {
            var path = Path.of(model);
            if (Files.exists(path)) tts = InflectTTS.load(path);
        }

        if (list && !explicitModel) {
            // --list without a model → show self-archive entries
            try {
                SelfArchive sa = SelfArchive.open();
                try {
                    System.out.printf("%-40s %10s%n", "Entry", "Size");
                    System.out.println("-----------------------------------------------");
                    for (SelfArchive.Entry e : sa.entries()) {
                        System.out.printf("%-40s %8.1f MB%n", e.name(), e.size() / 1e6);
                    }
                } finally {
                    sa.close();
                }
            } catch (IOException ignored) {
                System.err.println("no self-archive found");
            }
            return;
        }

        if (tts == null) {
            showUsage();
            return;
        }
        if (!overrides.isEmpty()) tts.setWordOverrides(overrides);

        if (list) {
            // tts is guaranteed non-null here (z:// handled above)
            var c = tts.config();
            System.out.printf(
                    "symbols=%d inter=%d hidden=%d filter=%d heads=%d layers=%d sr=%d"
                            + " initCh=%d%n",
                    c.symbolCount(),
                    c.interChannels(),
                    c.hiddenChannels(),
                    c.filterChannels(),
                    c.nHeads(),
                    c.nLayers(),
                    c.sampleRate(),
                    c.upsampleInitialChannel());
            long params = tts.weights().tensors().values().stream().mapToLong(t -> t.size()).sum();
            System.out.printf("tensors=%d params=%d%n", tts.weights().tensors().size(), params);
            return;
        }

        if (play) {
            tts.play(text, speed, variation, seed);
        } else {
            long t0 = System.currentTimeMillis();
            var audio = tts.synthesize(text, speed, variation, seed);
            long dt = System.currentTimeMillis() - t0;
            float max = 0;
            double rms = 0;
            var pcm = audio.pcm();
            for (float f : pcm) {
                max = Math.max(max, Math.abs(f));
                rms += f * f;
            }
            double rtf = pcm.length / (double) audio.sampleRate() / (dt / 1000.0);
            System.out.printf(
                    "synthesis: %d ms, %d samples (%.2f s), max=%.4f, rms=%.6f, %.2f×"
                            + " realtime%n",
                    dt,
                    pcm.length,
                    pcm.length / (float) audio.sampleRate(),
                    max,
                    Math.sqrt(rms / pcm.length),
                    rtf);
            AudioIO.writeWav(pcm, audio.sampleRate(), Path.of(out));
            System.out.println("wrote " + Path.of(out).toAbsolutePath());
        }
    }
}
