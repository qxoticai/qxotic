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

    @Test
    void appendGrowsWithoutRewriting() throws Exception {
        byte[] seed = {7};
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();
        long[] a = fp(12, 100);

        // create via appendTo on a missing file (delegates to freeze)
        Path file = Files.createTempFile("append", ".jkv");
        Files.delete(file);
        file.toFile().deleteOnExit();
        PromptCache<BlockResumeTest.FakeState> first =
                new PromptCache<>(codec, CacheStore.inMemory(), 1 << 20, seed);
        BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
        PromptCache<BlockResumeTest.FakeState>.Block tip = first.resume(new long[0], 0, s);
        s.ingestTo(12);
        first.commit(tip, a, 0, 12, s);
        first.appendTo(file);
        long size1 = Files.size(file);
        byte[] blobA = new byte[(int) codec.blockBytes(12)];
        try (var ch = java.nio.channels.FileChannel.open(file)) {
            ch.read(java.nio.ByteBuffer.wrap(blobA), FrozenBlocks.HEADER_BYTES);
        }

        // append prompt B (shares the first 12 as prefix, adds an 8-position tail)
        long[] b = java.util.Arrays.copyOf(a, 20);
        for (int i = 12; i < 20; i++) b[i] = 700 + i;
        PromptCache<BlockResumeTest.FakeState> grow =
                new PromptCache<>(
                        codec, CacheStore.inMemory(), 1 << 20, seed, FrozenBlocks.open(file, seed));
        BlockResumeTest.FakeState g = new BlockResumeTest.FakeState();
        PromptCache<BlockResumeTest.FakeState>.Block gt = grow.resume(b, 20, g);
        assertEquals(12, g.position, "append pass reuses the frozen prefix");
        g.ingestTo(20);
        grow.commit(gt, b, 12, 8, g);
        grow.appendTo(file);
        long size2 = Files.size(file);
        // growth = tail blob + index bytes + alignment; block A's stored bytes are UNTOUCHED
        assertTrue(
                size2 - size1 <= 512,
                "append cost is the new tail + index, not the catalog ("
                        + (size2 - size1)
                        + " bytes)");
        byte[] blobAAfter = new byte[blobA.length];
        try (var ch = java.nio.channels.FileChannel.open(file)) {
            ch.read(java.nio.ByteBuffer.wrap(blobAAfter), FrozenBlocks.HEADER_BYTES);
        }
        assertTrue(
                java.util.Arrays.equals(blobA, blobAAfter),
                "existing blob bytes are byte-identical after append (no rewrite)");

        // reopen: both prompts serve, content-exact
        FrozenBlocks reopened = FrozenBlocks.open(file, seed);
        assertEquals(2, reopened.blockCount());
        for (long[] prompt : new long[][] {a, b}) {
            PromptCache<BlockResumeTest.FakeState> serve =
                    new PromptCache<>(codec, CacheStore.inMemory(), 0, seed, reopened);
            BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
            serve.resume(prompt, prompt.length, r);
            assertEquals(prompt.length, r.position, "chain of " + prompt.length + " serves");
            for (int px = 0; px < prompt.length; px++)
                assertEquals(BlockResumeTest.FakeState.rowAt(px), r.rows[px]);
        }

        // crash simulation: old header + torn tail (append written, header flip lost)
        Path torn = Files.createTempFile("append-torn", ".jkv");
        torn.toFile().deleteOnExit();
        Files.copy(file, torn, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try (java.nio.channels.FileChannel ch =
                java.nio.channels.FileChannel.open(torn, java.nio.file.StandardOpenOption.WRITE)) {
            // restore the PRE-append header (count=1, indexOffset as after the first appendTo)
            java.nio.ByteBuffer flip =
                    java.nio.ByteBuffer.allocate(12).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            long firstIndexOffset =
                    FrozenBlocks.align(FrozenBlocks.HEADER_BYTES + codec.blockBytes(12));
            flip.putInt(1).putLong(firstIndexOffset).flip();
            ch.write(flip, FrozenBlocks.COUNT_OFFSET);
        }
        FrozenBlocks recovered = FrozenBlocks.open(torn, seed);
        assertEquals(1, recovered.blockCount(), "torn append: the old catalog is intact");
        PromptCache<BlockResumeTest.FakeState> serve =
                new PromptCache<>(codec, CacheStore.inMemory(), 0, seed, recovered);
        BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
        serve.resume(a, 12, r);
        assertEquals(12, r.position, "torn append: old prompt still serves");
    }
}
