package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * Who frees the weights, and when. An engine that loaded its own weights must free them at close;
 * one handed a {@link LoadedModel} must not, because the arena those weights live in belongs to
 * whoever loaded them and may hold more than this engine.
 *
 * <p>Fixture-gated: this needs real weights in a real arena, since the whole point is observing
 * whether that arena is still alive afterwards. It lives here rather than in jinfer-chat because
 * the ports depend on jinfer-chat, so that module has no architecture-dispatch provider to load
 * with.
 */
final class ChatEngineWeightsOwnershipTest {

    /** No block layer and no catalog: these tests are about the WEIGHTS arena. */
    private static final PromptCache.Options SESSIONS_ONLY =
            PromptCache.Options.DEFAULTS.withBlockBudget(0);

    private static Path model() {
        return TestModels.require("hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf");
    }

    @Test
    void anEngineGivenAModelDoesNotFreeTheCallersWeights() throws IOException {
        try (Arena weights = Arenas.newCrossThread()) {
            LoadedModel<?> loaded = Models.load(model(), weights);
            ChatEngine engine = new ChatEngine(loaded, "borrowed", SESSIONS_ONLY);

            assertSame(loaded, engine.loaded(), "the engine must use the model it was given");
            engine.close();

            assertTrue(
                    weights.scope().isAlive(),
                    "close() freed an arena the engine did not allocate");
            // and the weights are still readable, not merely 'alive'
            assertTrue(loaded.model().configuration().vocabularySize() > 0);
        }
    }

    @Test
    void anEngineThatLoadedItsOwnWeightsFreesThem() {
        ChatEngine engine = new ChatEngine(model(), null, SESSIONS_ONLY);
        engine.close();
        // the arena is internal, so the observable contract is that close returns and stays
        // idempotent - a second close would hit the JDK's one-shot Arena.close if it were not
        engine.close();
    }

    @Test
    void closeIsIdempotentOnABorrowedEngineToo() throws IOException {
        try (Arena weights = Arenas.newCrossThread()) {
            ChatEngine engine =
                    new ChatEngine(Models.load(model(), weights), "borrowed", SESSIONS_ONLY);
            engine.close();
            engine.close();
            assertTrue(weights.scope().isAlive());
        }
    }

    @Test
    void aClosedArenaIsWhatABorrowedEngineWillNotSaveYouFrom() throws IOException {
        // the law the code cannot enforce, pinned as documentation: closing YOUR arena while the
        // engine still lives is on you. This only asserts the ordering the contract prescribes -
        // engine first, then the arena - leaves nothing alive.
        Arena weights = Arenas.newCrossThread();
        ChatEngine engine =
                new ChatEngine(Models.load(model(), weights), "borrowed", SESSIONS_ONLY);
        engine.close();
        weights.close();
        assertFalse(weights.scope().isAlive());
    }
}
