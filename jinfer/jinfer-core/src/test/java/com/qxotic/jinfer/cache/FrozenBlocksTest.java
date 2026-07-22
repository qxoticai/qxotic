package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.CacheStore;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * The frozen-artifact laws: freeze/open round-trips content exactly (verified through the
 * BlockResumeTest fake codec's readable rows/residue); a wrong model seed is rejected with a clear
 * error; a writable cache grafted over the artifact matches through frozen blocks, dedups commits
 * against them, grows past them, and NEVER evicts or frees them under budget pressure.
 */
public final class FrozenBlocksTest {

    static long[] fp(int n, long base) {
        long[] fp = new long[n];
        for (int i = 0; i < n; i++) fp[i] = base + i;
        return fp;
    }

    @Test
    void freezeOpenOverlayAndEvictionIsolation() throws Exception {
        byte[] seed = {42};
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();

        // compile time: two prompts sharing a 10-position prefix, frozen into one artifact
        PromptCache<BlockResumeTest.FakeState> build =
                new PromptCache<>(codec, CacheStore.inMemory(), 1 << 20, seed);
        long[] a = fp(16, 100); // prompt A: [shared 10][A tail 6]
        long[] b = fp(16, 100); // prompt B: [shared 10][B tail 6]
        for (int i = 10; i < 16; i++) b[i] = 900 + i;
        for (long[] prompt : new long[][] {a, b}) {
            BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
            PromptCache<BlockResumeTest.FakeState>.Block tip = build.resume(new long[0], 0, s);
            s.ingestTo(10);
            tip = build.commit(tip, prompt, 0, 10, s); // shared prefix block (dedups on B)
            s.ingestTo(16);
            tip = build.commit(tip, prompt, 10, 6, s);
        }
        assertTrue(build.stats().contains("blocks=3"), "shared prefix stored once");
        Path file = Files.createTempFile("frozen", ".jkv");
        file.toFile().deleteOnExit();
        build.freeze(file);

        // serve time: wrong seed rejected, right seed opens
        assertThrows(IllegalStateException.class, () -> FrozenBlocks.open(file, new byte[] {9}));
        FrozenBlocks frozen = FrozenBlocks.open(file, seed);
        assertEquals(3, frozen.blockCount());

        // a TINY writable cache over the frozen base (budget fits ~2 own blocks)
        PromptCache<BlockResumeTest.FakeState> live =
                new PromptCache<>(codec, CacheStore.inMemory(), 200, seed, frozen);

        // both frozen prompts resume fully, content-exact
        for (long[] prompt : new long[][] {a, b}) {
            BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
            live.resume(prompt, 16, r);
            assertEquals(16, r.position, "frozen prompt resumes fully");
            for (int p = 0; p < 16; p++)
                assertEquals(BlockResumeTest.FakeState.rowAt(p), r.rows[p], "row " + p);
            assertEquals(BlockResumeTest.FakeState.residueAt(16), r.residue);
        }

        // grow past the frozen chain, evict under pressure, frozen blocks stay servable
        for (int round = 0; round < 6; round++) {
            BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
            PromptCache<BlockResumeTest.FakeState>.Block tip = live.resume(a, 16, s);
            assertEquals(16, s.position, "frozen prefix hit on round " + round);
            s.ingestTo(30);
            long[] grown = java.util.Arrays.copyOf(a, 30);
            for (int i = 16; i < 30; i++) grown[i] = 5000 + round * 100 + i; // diverging tails
            live.commit(tip, grown, 16, 14, s);
        }
        // budget only fits ~2 grown tails: earlier tails were evicted - but never frozen blocks
        BlockResumeTest.FakeState check = new BlockResumeTest.FakeState();
        live.resume(b, 16, check);
        assertEquals(16, check.position, "frozen blocks survive eviction pressure");

        // corruption: flip one KV byte in the artifact - the CRC gate turns it into a MISS,
        // never a wrong restore (open a separate copy so the mmap above stays pristine)
        Path corrupt = Files.createTempFile("frozen-corrupt", ".jkv");
        corrupt.toFile().deleteOnExit();
        Files.copy(file, corrupt, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try (java.nio.channels.FileChannel ch =
                java.nio.channels.FileChannel.open(
                        corrupt, java.nio.file.StandardOpenOption.WRITE)) {
            ch.write(
                    java.nio.ByteBuffer.wrap(new byte[] {(byte) 0xAA}),
                    FrozenBlocks.HEADER_BYTES + 3);
        }
        PromptCache<BlockResumeTest.FakeState> corrupted =
                new PromptCache<>(
                        codec,
                        CacheStore.inMemory(),
                        1 << 20,
                        seed,
                        FrozenBlocks.open(corrupt, seed));
        BlockResumeTest.FakeState cr = new BlockResumeTest.FakeState();
        corrupted.resume(a, 16, cr);
        assertEquals(0, cr.position, "corrupted frozen block degrades to a miss, never restores");

        // commit dedup against a frozen block: re-ingesting prompt A stores nothing new
        String before = live.stats().replaceAll(" hits=.*", "");
        BlockResumeTest.FakeState again = new BlockResumeTest.FakeState();
        PromptCache<BlockResumeTest.FakeState>.Block tip = live.resume(new long[0], 0, again);
        again.ingestTo(10);
        tip = live.commit(tip, a, 0, 10, again);
        assertEquals(
                before,
                live.stats().replaceAll(" hits=.*", ""),
                "commit onto a frozen chain dedups: same blocks, same bytes");
        assertTrue(tip.frozen, "the deduped tip IS the frozen block");
    }
}
