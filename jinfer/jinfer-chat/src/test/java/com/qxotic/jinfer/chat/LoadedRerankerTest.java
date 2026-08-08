package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Model;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The reranking serving contract, model-free: what {@link LoadedReranker#scoreAll} ingests, where
 * the cursor sits when each candidate lands, and what it refuses. The fakes record every forward,
 * so the frame-reuse law ("the judge frame is prefilled exactly once per call") is asserted on the
 * token stream itself rather than inferred from timings.
 */
class LoadedRerankerTest {

    // ---- fakes: a state that only tracks the cursor, a model that records what it was fed ----

    static final class FakeState extends BaseState {
        final List<int[]> forwarded = new ArrayList<>();
        final List<Integer> forwardedAt = new ArrayList<>(); // cursor BEFORE each forward
        int contextCapacity = 1024;
        int batchCapacity = 512;

        FakeState() {
            super(Arena.ofAuto());
        }

        @Override
        public int contextCapacity() {
            return contextCapacity;
        }

        @Override
        public int batchCapacity() {
            return batchCapacity;
        }

        @Override
        public void reset() {
            resumeAt(0);
        }
    }

    static final class FakeModel implements Model<Config, Object, FakeState> {
        @Override
        public Config config() {
            return null;
        }

        @Override
        public Object weights() {
            return null;
        }

        @Override
        public FakeState newState(int contextCapacity, int batchCapacity, Arena arena) {
            return new FakeState();
        }

        @Override
        public void forward(FakeState s, Batch batch) {
            s.forwardedAt.add(s.position());
            s.forwarded.add(((Batch.Input.Tokens) batch.input()).ids());
            s.advance(batch.count(), batch.outputs());
        }
    }

    /** Frame = 5 tokens of 1s; each document = its own length of 2s; score = the cursor it saw. */
    static final class FakeReranker implements Reranker.CrossEncoder<FakeState> {
        final FakeModel model = new FakeModel();

        @Override
        public com.qxotic.jinfer.Model<?, ?, FakeState> model() {
            return model;
        }

        final List<Integer> scoredAt = new ArrayList<>();

        @Override
        public String defaultInstruction() {
            return "judge";
        }

        @Override
        public Batch head(String instruction, String query) {
            return Batch.prefill(fill(5, 1));
        }

        @Override
        public Batch document(String document) {
            return Batch.prefill(fill(document.length(), 2));
        }

        @Override
        public double score(FakeState state) {
            scoredAt.add(state.position());
            return state.position() / 100.0; // distinct per candidate, ordering-checkable
        }

        private static int[] fill(int n, int value) {
            int[] ids = new int[n];
            java.util.Arrays.fill(ids, value);
            return ids;
        }
    }

    private static LoadedReranker<FakeState> loaded(FakeReranker reranker) {
        return new LoadedReranker<>(reranker.model, reranker, "fake.gguf");
    }

    private static List<Double> scoreAll(
            LoadedReranker<FakeState> loaded, FakeState state, List<String> documents) {
        List<Double> scores = new ArrayList<>();
        loaded.scoreAll(state, "judge", "q", documents, scores::add);
        return scores;
    }

    // ---- the laws ----

    @Test
    void framePrefilledOnceThenOneTailPerCandidate() {
        FakeState state = new FakeState();
        scoreAll(loaded(new FakeReranker()), state, List.of("aa", "bbb", "cccc"));
        assertEquals(4, state.forwarded.size(), "one frame + one tail per candidate");
        assertEquals(5, state.forwarded.get(0).length, "the frame");
        assertEquals(1, state.forwarded.get(0)[0], "frame tokens");
        assertEquals(2, state.forwarded.get(1).length, "candidate 'aa'");
        assertEquals(3, state.forwarded.get(2).length, "candidate 'bbb'");
        assertEquals(4, state.forwarded.get(3).length, "candidate 'cccc'");
    }

    @Test
    void everyCandidateStartsAtTheFrameCursor() {
        FakeState state = new FakeState();
        scoreAll(loaded(new FakeReranker()), state, List.of("aa", "bbb", "cccc"));
        // the whole point of the rewind: candidate N never sees candidate N-1's tokens
        assertEquals(List.of(0, 5, 5, 5), state.forwardedAt);
    }

    @Test
    void verdictIsReadPerCandidateAfterItLands() {
        FakeState state = new FakeState();
        FakeReranker reranker = new FakeReranker();
        List<Double> scores = scoreAll(loaded(reranker), state, List.of("aa", "bbb"));
        assertEquals(List.of(7, 8), reranker.scoredAt, "frame(5) + candidate, per candidate");
        assertEquals(List.of(0.07, 0.08), scores, "one score per candidate, in input order");
    }

    @Test
    void countsEveryIngestedToken() {
        FakeState state = new FakeState();
        int total =
                loaded(new FakeReranker())
                        .scoreAll(state, "judge", "q", List.of("aa", "bbb"), s -> {});
        assertEquals(5 + 2 + 3, total, "frame counted ONCE, then every candidate");
    }

    @Test
    void resetsTheStateItIsHanded() {
        FakeState state = new FakeState();
        state.resumeAt(99); // a state left dirty by a previous call
        scoreAll(loaded(new FakeReranker()), state, List.of("aa"));
        assertEquals(0, state.forwardedAt.get(0), "the frame must land at position 0");
    }

    @Test
    void noCandidatesCostsNothing() {
        FakeState state = new FakeState();
        int total = loaded(new FakeReranker()).scoreAll(state, "judge", "q", List.of(), s -> {});
        assertEquals(0, total);
        assertTrue(state.forwarded.isEmpty(), "an empty retrieval must not prefill a frame");
    }

    @Test
    void oversizedCandidateIsRefusedByIndex() {
        FakeState state = new FakeState();
        state.contextCapacity = 12; // frame(5) + 6 fits, + 8 does not
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                scoreAll(
                                        loaded(new FakeReranker()),
                                        state,
                                        List.of("aaaaaa", "bbbbbbbb")));
        assertTrue(e.getMessage().contains("document 1"), e.getMessage());
        assertTrue(e.getMessage().contains("contextLength"), e.getMessage());
    }

    @Test
    void batchesWiderThanTheStateAreChunked() {
        FakeState state = new FakeState();
        state.batchCapacity = 2; // the 5-token frame cannot ride in one forward
        scoreAll(loaded(new FakeReranker()), state, List.of("aa"));
        assertEquals(4, state.forwarded.size(), "frame chunked 2+2+1, then the candidate");
        assertEquals(List.of(0, 2, 4, 5), state.forwardedAt);
    }
}
