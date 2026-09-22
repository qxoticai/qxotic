package com.qxotic.jinfer.models.parakeet;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * End to end against the goldens ({@code golden.tsv}): real speech in four languages and at two
 * sample rates, through the public API, for every quantization: offline, streamed, long-form across
 * chunk commits, on a reused state, under gain and silence, and from parallel states. Tolerant by
 * design (see {@link Golden}): quantization and float drift pass, regressions do not.
 */
@Tag("integration")
class ParakeetGoldenTest {

    private static final String[] MODELS = {
        "tdt-0.6b-v3-f16", "tdt-0.6b-v3-q8_0", "tdt-0.6b-v3-q4_k"
    };

    /** The model the scenario tests run on: the default deployment quant. */
    private static final String SCENARIO_MODEL = "tdt-0.6b-v3-q8_0";

    private static final int RATE = 16_000;
    private static Arena arena;
    private static final Map<String, Parakeet> MODELS_LOADED = new ConcurrentHashMap<>();

    @BeforeAll
    static void open() {
        arena = Arena.ofShared();
    }

    @AfterAll
    static void close() {
        if (arena != null) arena.close();
    }

    private static Parakeet model(String name) {
        return MODELS_LOADED.computeIfAbsent(
                name,
                n -> {
                    Path file = TestModels.require("mudler/parakeet-cpp-gguf/" + n + ".gguf");
                    try {
                        return Fixtures.load(file, arena);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                });
    }

    static Stream<Arguments> modelsAndClips() {
        return Stream.of(MODELS)
                .flatMap(model -> Golden.clips().stream().map(clip -> Arguments.of(model, clip)));
    }

    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("modelsAndClips")
    void transcribesLikeTheGolden(String model, String clip) {
        float[] pcm = TestAudio.clip(clip);
        Golden.of(clip).assertHeardIn(model(model).transcribe(pcm), seconds(pcm), 0);
    }

    /** Fed in random chunks with partials polled in between, a stream ends at the golden. */
    @Test
    void aStreamFedInPiecesMatchesTheGolden() {
        Parakeet parakeet = model(SCENARIO_MODEL);
        Random random = new Random(42);
        for (String clip : Golden.clips()) {
            float[] pcm = TestAudio.clip(clip);
            Transcription streamed = Fixtures.streamed(parakeet, pcm, random, 12_000, 4);
            Golden.of(clip).assertHeardIn(streamed, seconds(pcm), 0);
        }
    }

    /**
     * The English clips back to back with a second of silence between them, in 30 s chunks so
     * several commit: the stitched transcript is the stitched golden, words in place.
     */
    @Test
    void longFormAcrossChunkCommitsMatchesTheGoldens() {
        List<String> clips = List.of("en", "en0", "ls0", "ls1", "ls8k");
        double[] offsets = new double[clips.size()];
        float[] joined = joined(clips, offsets, false);
        List<Golden> goldens = clips.stream().map(Golden::of).toList();
        try (var chunks = Fixtures.chunkSeconds(30)) {
            try (Parakeet.State state = model(SCENARIO_MODEL).newState()) {
                Golden.concat(goldens, offsets)
                        .assertHeardIn(
                                model(SCENARIO_MODEL).transcribe(state, joined),
                                seconds(joined),
                                0);
            }
        }
    }

    /**
     * The same clips streamed live, a chunk final every 2 s: the stitched golden again. The gaps
     * hold room noise (-60 dBFS) instead of exact zeros, as live audio does: v3 misreads windows
     * this short around digital silence (NVIDIA-NeMo/Speech#15757).
     */
    @Test
    void longFormStreamedLiveMatchesTheGoldens() {
        List<String> clips = List.of("en", "en0", "ls0", "ls1", "ls8k");
        double[] offsets = new double[clips.size()];
        float[] joined = joined(clips, offsets, true);
        List<Golden> goldens = clips.stream().map(Golden::of).toList();
        Transcription streamed =
                Fixtures.streamed(model(SCENARIO_MODEL), joined, new Random(5), 1_600, 0);
        Golden.concat(goldens, offsets).assertHeardIn(streamed, seconds(joined), 0);
    }

    /**
     * Across chunk commits, the transcript does not depend on how the audio arrives: random pieces
     * with partials polled in between give the stream's tokens, text and timings exactly.
     */
    @Test
    void longFormStreamedInPiecesIsTheSameTranscript() {
        Parakeet parakeet = model(SCENARIO_MODEL);
        float[] pcm = joined(List.of("en", "en0", "ls0", "ls1", "ls8k"), null, true);
        Fixtures.assertSameTranscript(
                Fixtures.streamedWhole(parakeet, pcm),
                Fixtures.streamed(parakeet, pcm, new Random(3), 40_000, 8),
                "streamed");
    }

    /**
     * The clips back to back, a second of silence after each, exact zeros or room noise; {@code
     * offsets} gets their starts.
     */
    private static float[] joined(List<String> clips, double[] offsets, boolean roomNoise) {
        List<float[]> parts = clips.stream().map(TestAudio::clip).toList();
        float[] joined = new float[parts.stream().mapToInt(part -> part.length + RATE).sum()];
        if (roomNoise) {
            Random noise = new Random(1);
            for (int i = 0; i < joined.length; i++)
                joined[i] = (float) (1e-3 * noise.nextGaussian());
        }
        for (int i = 0, at = 0; i < parts.size(); at += parts.get(i).length + RATE, i++) {
            if (offsets != null) offsets[i] = (double) at / RATE;
            System.arraycopy(parts.get(i), 0, joined, at, parts.get(i).length);
        }
        return joined;
    }

    /**
     * One state serves every clip, twice over: nothing leaks from one transcription to the next.
     */
    @Test
    void oneStateServesEveryClipTwice() {
        Parakeet parakeet = model(SCENARIO_MODEL);
        try (Parakeet.State state = parakeet.newState()) {
            for (int pass = 0; pass < 2; pass++) {
                for (String clip : Golden.clips()) {
                    float[] pcm = TestAudio.clip(clip);
                    Golden.of(clip).assertHeardIn(parakeet.transcribe(state, pcm), seconds(pcm), 0);
                }
            }
        }
    }

    /**
     * Per-feature normalization cancels gain, and leading silence only shifts time: quieter, louder
     * (unclipped) and delayed speech all say the golden, the delayed words shifted.
     */
    @Test
    void gainAndLeadingSilenceKeepTheGolden() {
        Parakeet parakeet = model(SCENARIO_MODEL);
        float[] pcm = TestAudio.clip("en0");
        float peak = 0;
        for (float sample : pcm) peak = Math.max(peak, Math.abs(sample));
        for (float gain : new float[] {0.25f, 0.95f / peak}) {
            float[] scaled = pcm.clone();
            for (int i = 0; i < scaled.length; i++) scaled[i] *= gain;
            Golden.of("en0").assertHeardIn(parakeet.transcribe(scaled), seconds(scaled), 0);
        }
        int silence = 20_480; // 1.28 s: 16 encoder frames exactly
        float[] delayed = new float[silence + pcm.length];
        System.arraycopy(pcm, 0, delayed, silence, pcm.length);
        Golden.of("en0")
                .assertHeardIn(
                        parakeet.transcribe(delayed), seconds(delayed), (double) silence / RATE);
    }

    /** States are independent pipelines: four threads, four states, every clip heard right. */
    @Test
    void parallelStatesEachMatchTheGolden() throws Exception {
        Parakeet parakeet = model(SCENARIO_MODEL);
        List<String> clips = Golden.clips();
        try (ExecutorService pool = Executors.newFixedThreadPool(4)) {
            List<Future<?>> runs = new ArrayList<>();
            for (int worker = 0; worker < 4; worker++) {
                int first = worker;
                runs.add(
                        pool.submit(
                                () -> {
                                    try (Parakeet.State state = parakeet.newState()) {
                                        for (int i = first; i < clips.size(); i += 4) {
                                            float[] pcm = TestAudio.clip(clips.get(i));
                                            Golden.of(clips.get(i))
                                                    .assertHeardIn(
                                                            parakeet.transcribe(state, pcm),
                                                            seconds(pcm),
                                                            0);
                                        }
                                    }
                                    return null;
                                }));
            }
            for (Future<?> run : runs) run.get(); // rethrows any assertion from its thread
        }
    }

    private static double seconds(float[] pcm) {
        return (double) pcm.length / RATE;
    }
}
