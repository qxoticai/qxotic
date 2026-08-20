package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Random;
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
        ContentKey seed = ContentKey.sha256(new byte[] {42});
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();

        // compile time: two prompts sharing a 10-position prefix, frozen into one artifact
        BlockTree<BlockResumeTest.FakeState> build =
                new BlockTree<>(codec, CacheStore.inMemory(), 1 << 20, seed);
        long[] a = fp(16, 100); // prompt A: [shared 10][A tail 6]
        long[] b = fp(16, 100); // prompt B: [shared 10][B tail 6]
        for (int i = 10; i < 16; i++) b[i] = 900 + i;
        for (long[] prompt : new long[][] {a, b}) {
            BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
            BlockTree<BlockResumeTest.FakeState>.Block tip = build.resume(new long[0], 0, s);
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
        assertThrows(
                IllegalStateException.class,
                () -> FrozenBlocks.open(file, ContentKey.sha256(new byte[] {9})));
        FrozenBlocks frozen = FrozenBlocks.open(file, seed);
        assertEquals(3, frozen.blockCount());

        // a TINY writable cache over the frozen base (budget fits ~2 own blocks)
        BlockTree<BlockResumeTest.FakeState> live =
                new BlockTree<>(codec, CacheStore.inMemory(), 200, seed, frozen);

        // both frozen prompts resume fully, content-exact
        for (long[] prompt : new long[][] {a, b}) {
            BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
            live.resume(prompt, 16, r);
            assertEquals(16, r.position(), "frozen prompt resumes fully");
            for (int p = 0; p < 16; p++)
                assertEquals(BlockResumeTest.FakeState.rowAt(p), r.rows[p], "row " + p);
            assertEquals(BlockResumeTest.FakeState.residueAt(16), r.residue);
        }

        // grow past the frozen chain, evict under pressure, frozen blocks stay servable
        for (int round = 0; round < 6; round++) {
            BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
            BlockTree<BlockResumeTest.FakeState>.Block tip = live.resume(a, 16, s);
            assertEquals(16, s.position(), "frozen prefix hit on round " + round);
            s.ingestTo(30);
            long[] grown = Arrays.copyOf(a, 30);
            for (int i = 16; i < 30; i++) grown[i] = 5000 + round * 100 + i; // diverging tails
            live.commit(tip, grown, 16, 14, s);
        }
        // budget only fits ~2 grown tails: earlier tails were evicted - but never frozen blocks
        BlockResumeTest.FakeState check = new BlockResumeTest.FakeState();
        live.resume(b, 16, check);
        assertEquals(16, check.position(), "frozen blocks survive eviction pressure");

        // corruption: flip one KV byte in the artifact - the CRC gate turns it into a MISS,
        // never a wrong restore (open a separate copy so the mmap above stays pristine)
        Path corrupt = Files.createTempFile("frozen-corrupt", ".jkv");
        corrupt.toFile().deleteOnExit();
        Files.copy(file, corrupt, StandardCopyOption.REPLACE_EXISTING);
        try (FileChannel ch = FileChannel.open(corrupt, StandardOpenOption.WRITE)) {
            ch.write(ByteBuffer.wrap(new byte[] {(byte) 0xAA}), FrozenBlocks.HEADER_BYTES + 3);
        }
        BlockTree<BlockResumeTest.FakeState> corrupted =
                new BlockTree<>(
                        codec,
                        CacheStore.inMemory(),
                        1 << 20,
                        seed,
                        FrozenBlocks.open(corrupt, seed));
        BlockResumeTest.FakeState cr = new BlockResumeTest.FakeState();
        corrupted.resume(a, 16, cr);
        assertEquals(0, cr.position(), "corrupted frozen block degrades to a miss, never restores");

        // commit dedup against a frozen block: re-ingesting prompt A stores nothing new
        String before = live.stats().replaceAll(" hits=.*", "");
        BlockResumeTest.FakeState again = new BlockResumeTest.FakeState();
        BlockTree<BlockResumeTest.FakeState>.Block tip = live.resume(new long[0], 0, again);
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
        ContentKey seed = ContentKey.sha256(new byte[] {7});
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();
        long[] a = fp(12, 100);

        // create via appendTo on a missing file (delegates to freeze)
        Path file = Files.createTempFile("append", ".jkv");
        Files.delete(file);
        file.toFile().deleteOnExit();
        BlockTree<BlockResumeTest.FakeState> first =
                new BlockTree<>(codec, CacheStore.inMemory(), 1 << 20, seed);
        BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
        BlockTree<BlockResumeTest.FakeState>.Block tip = first.resume(new long[0], 0, s);
        s.ingestTo(12);
        first.commit(tip, a, 0, 12, s);
        first.appendTo(file);
        long size1 = Files.size(file);
        byte[] blobA = new byte[(int) codec.byteSize(12)];
        try (var ch = FileChannel.open(file)) {
            ch.read(ByteBuffer.wrap(blobA), FrozenBlocks.HEADER_BYTES);
        }

        // append prompt B (shares the first 12 as prefix, adds an 8-position tail)
        long[] b = Arrays.copyOf(a, 20);
        for (int i = 12; i < 20; i++) b[i] = 700 + i;
        BlockTree<BlockResumeTest.FakeState> grow =
                new BlockTree<>(
                        codec, CacheStore.inMemory(), 1 << 20, seed, FrozenBlocks.open(file, seed));
        BlockResumeTest.FakeState g = new BlockResumeTest.FakeState();
        BlockTree<BlockResumeTest.FakeState>.Block gt = grow.resume(b, 20, g);
        assertEquals(12, g.position(), "append pass reuses the frozen prefix");
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
        try (var ch = FileChannel.open(file)) {
            ch.read(ByteBuffer.wrap(blobAAfter), FrozenBlocks.HEADER_BYTES);
        }
        assertTrue(
                Arrays.equals(blobA, blobAAfter),
                "existing blob bytes are byte-identical after append (no rewrite)");

        // reopen: both prompts serve, content-exact
        FrozenBlocks reopened = FrozenBlocks.open(file, seed);
        assertEquals(2, reopened.blockCount());
        for (long[] prompt : new long[][] {a, b}) {
            BlockTree<BlockResumeTest.FakeState> serve =
                    new BlockTree<>(codec, CacheStore.inMemory(), 0, seed, reopened);
            BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
            serve.resume(prompt, prompt.length, r);
            assertEquals(prompt.length, r.position(), "chain of " + prompt.length + " serves");
            for (int px = 0; px < prompt.length; px++)
                assertEquals(BlockResumeTest.FakeState.rowAt(px), r.rows[px]);
        }

        // THIRD boot: mount the twice-grown catalog, append again - the accumulating loop that
        // a server's repeated restarts drive (a stale indexOffset/count bug surfaces exactly here)
        long[] c = Arrays.copyOf(a, 15);
        for (int i = 12; i < 15; i++) c[i] = 900 + i;
        BlockTree<BlockResumeTest.FakeState> third =
                new BlockTree<>(
                        codec, CacheStore.inMemory(), 1 << 20, seed, FrozenBlocks.open(file, seed));
        BlockResumeTest.FakeState t = new BlockResumeTest.FakeState();
        BlockTree<BlockResumeTest.FakeState>.Block tt = third.resume(c, 15, t);
        assertEquals(12, t.position(), "third boot reuses the shared prefix");
        t.ingestTo(15);
        third.commit(tt, c, 12, 3, t);
        third.appendTo(file);
        assertEquals(3, FrozenBlocks.open(file, seed).blockCount(), "three boots, three blocks");

        // SINGLE-WRITER LAW: a mount whose view went stale (another writer appended since) must
        // refuse loudly instead of overwriting the other writer's blocks
        BlockTree<BlockResumeTest.FakeState> stale =
                new BlockTree<>(
                        codec, CacheStore.inMemory(), 1 << 20, seed, FrozenBlocks.open(file, seed));
        BlockResumeTest.FakeState st = new BlockResumeTest.FakeState();
        long[] d = Arrays.copyOf(a, 13);
        d[12] = 4242;
        BlockTree<BlockResumeTest.FakeState>.Block sTip = stale.resume(d, 13, st);
        st.ingestTo(13);
        stale.commit(sTip, d, 12, 1, st);
        // another writer grows the file after `stale` mounted
        BlockTree<BlockResumeTest.FakeState> rival =
                new BlockTree<>(
                        codec, CacheStore.inMemory(), 1 << 20, seed, FrozenBlocks.open(file, seed));
        BlockResumeTest.FakeState rv = new BlockResumeTest.FakeState();
        long[] e = Arrays.copyOf(a, 13);
        e[12] = 5353;
        BlockTree<BlockResumeTest.FakeState>.Block rTip = rival.resume(e, 13, rv);
        rv.ingestTo(13);
        rival.commit(rTip, e, 12, 1, rv);
        rival.appendTo(file);
        var refused = assertThrows(IOException.class, () -> stale.appendTo(file));
        assertTrue(
                refused.getMessage().contains("another writer"),
                "stale append must refuse, not overwrite: " + refused.getMessage());
        assertEquals(
                4,
                FrozenBlocks.open(file, seed).blockCount(),
                "the rival's append survives untouched");

        // crash simulation: old header + torn tail (append written, header flip lost)
        Path torn = Files.createTempFile("append-torn", ".jkv");
        torn.toFile().deleteOnExit();
        Files.copy(file, torn, StandardCopyOption.REPLACE_EXISTING);
        try (FileChannel ch = FileChannel.open(torn, StandardOpenOption.WRITE)) {
            // restore the PRE-append header (count=1, indexOffset as after the first appendTo)
            ByteBuffer flip = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
            long firstIndexOffset =
                    FrozenBlocks.align(FrozenBlocks.HEADER_BYTES + codec.byteSize(12));
            flip.putInt(1).putLong(firstIndexOffset).flip();
            ch.write(flip, FrozenBlocks.COUNT_OFFSET);
        }
        FrozenBlocks recovered = FrozenBlocks.open(torn, seed);
        assertEquals(1, recovered.blockCount(), "torn append: the old catalog is intact");
        BlockTree<BlockResumeTest.FakeState> serve =
                new BlockTree<>(codec, CacheStore.inMemory(), 0, seed, recovered);
        BlockResumeTest.FakeState r = new BlockResumeTest.FakeState();
        serve.resume(a, 12, r);
        assertEquals(12, r.position(), "torn append: old prompt still serves");
    }

    @Test
    void corruptDeepChainDiscardsIterativelyNeverStackOverflows() throws Exception {
        int depth = 65_536, victimDepth = depth / 2; // recursion over this depth would SO
        ContentKey seed = ContentKey.sha256(new byte[] {43});
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();

        // one block per position: chain depth == depth
        BlockTree<BlockResumeTest.FakeState> build =
                new BlockTree<>(codec, CacheStore.inMemory(), 1L << 28, seed);
        long[] fp = fp(depth, 1000);
        BlockResumeTest.FakeState w = new BlockResumeTest.FakeState(depth);
        BlockTree<BlockResumeTest.FakeState>.Block tip = build.resume(new long[0], 0, w);
        for (int p = 0; p < depth; p++) {
            w.ingestTo(p + 1);
            tip = build.commit(tip, fp, p, 1, w);
        }
        Path file = Files.createTempFile("frozen-deep", ".jkv");
        file.toFile().deleteOnExit();
        build.freeze(file);

        // corrupt the victim's blob BEFORE any restore verifies it (frozenVerified memoizes);
        // in a linear chain the BFS index IS the depth
        FrozenBlocks opened = FrozenBlocks.open(file, seed);
        long victimOffset = opened.entries().get(victimDepth - 1).offset();
        try (FileChannel ch =
                FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            ByteBuffer one = ByteBuffer.allocate(1);
            ch.read(one, victimOffset);
            one.flip();
            one.put(0, (byte) (one.get(0) ^ 0xFF));
            ch.write(one, victimOffset);
        }

        BlockTree<BlockResumeTest.FakeState> mounted =
                new BlockTree<>(codec, CacheStore.inMemory(), 1L << 28, seed, opened);
        BlockResumeTest.FakeState r = new BlockResumeTest.FakeState(depth);
        mounted.resume(fp, depth, r);
        assertEquals(
                victimDepth - 1,
                r.position(),
                "the corrupt block's subtree is discarded; the verified prefix survives");
        assertEquals(
                depth - victimDepth + 1, // the victim itself plus everything chained on it
                mounted.sample().discards(),
                "every block chained on the corrupt one is discarded, once");

        // the surviving prefix serves cleanly afterwards
        BlockResumeTest.FakeState again = new BlockResumeTest.FakeState(depth);
        mounted.resume(fp, depth, again);
        assertEquals(victimDepth - 1, again.position());
        assertEquals(BlockResumeTest.FakeState.residueAt(victimDepth - 1), again.residue);
    }

    @Test
    void corruptArtifactsFailWithOneStableError() throws Exception {
        ContentKey seed = ContentKey.sha256(new byte[] {42});
        Path file = artifactWithTwoBlocks(seed);
        byte[] pristine = Files.readAllBytes(file);
        int index = (int) indexOffset(pristine);

        // header: negative count, absurd count (index past EOF), index offset out of range
        assertCorrupt(mutateInt(pristine, FrozenBlocks.COUNT_OFFSET, -1), seed);
        assertCorrupt(mutateInt(pristine, FrozenBlocks.COUNT_OFFSET, 1 << 20), seed);
        assertCorrupt(mutateLong(pristine, FrozenBlocks.COUNT_OFFSET + 4, -1), seed);
        assertCorrupt(mutateLong(pristine, FrozenBlocks.COUNT_OFFSET + 4, 8), seed);
        assertCorrupt(
                mutateLong(pristine, FrozenBlocks.COUNT_OFFSET + 4, pristine.length + 64), seed);

        // entry 0: inverted span, negative from
        assertCorrupt(mutateInt(pristine, index + 64, -4), seed); // from
        assertCorrupt(mutateInt(pristine, index + 68, -1), seed); // to < from
        // entry 0: blob below the KV region, negative length, length past the index
        assertCorrupt(mutateLong(pristine, index + 72, 0), seed);
        assertCorrupt(mutateLong(pristine, index + 80, -8), seed);
        assertCorrupt(
                mutateLong(pristine, index + 80, index - FrozenBlocks.HEADER_BYTES + 1), seed);
        // entry 0: wrong parent (the seed-derived root zeroed out)
        byte[] wrongRoot = pristine.clone();
        for (int i = 32; i < 64; i++) wrongRoot[index + i] = 0;
        assertCorrupt(wrongRoot, seed);
        // entry 1 before its parent: swap the two 96-byte records (parents-first violated)
        byte[] swapped = pristine.clone();
        for (int i = 0; i < FrozenBlocks.INDEX_ENTRY_BYTES; i++) {
            byte t = swapped[index + i];
            swapped[index + i] = swapped[index + FrozenBlocks.INDEX_ENTRY_BYTES + i];
            swapped[index + FrozenBlocks.INDEX_ENTRY_BYTES + i] = t;
        }
        assertCorrupt(swapped, seed);
        // duplicate key: entry 1's key overwritten with entry 0's
        byte[] dup = pristine.clone();
        System.arraycopy(pristine, index, dup, index + 96, 32);
        assertCorrupt(dup, seed);
    }

    @Test
    void fuzzedArtifactsOpenOrFailStablyNeverIncidentally() throws Exception {
        ContentKey seed = ContentKey.sha256(new byte[] {42});
        byte[] pristine = Files.readAllBytes(artifactWithTwoBlocks(seed));
        Random random = new Random(20260816L);
        for (int trial = 0; trial < 500; trial++) {
            byte[] mutated = pristine.clone();
            for (int f = 0, flips = 1 + random.nextInt(3); f < flips; f++) {
                mutated[random.nextInt(mutated.length)] ^= (byte) (1 << random.nextInt(8));
            }
            Path tmp = Files.createTempFile("frozen-fuzz", ".jkv");
            tmp.toFile().deleteOnExit();
            Files.write(tmp, mutated);
            try {
                // may still open: KV-byte and structure-preserving flips stay servable
                // (blob CRCs are verified lazily at restore, by design)
                FrozenBlocks.open(tmp, seed);
            } catch (IllegalStateException stable) {
                // the one error a corrupt artifact fails with
            } catch (RuntimeException incidental) {
                throw new AssertionError(
                        "trial " + trial + " failed incidentally, not stably", incidental);
            }
        }
    }

    /** A valid two-block artifact: a 12-position block with a 3-position child. */
    private static Path artifactWithTwoBlocks(ContentKey seed) throws IOException {
        BlockResumeTest.FakeCodec codec = new BlockResumeTest.FakeCodec();
        BlockTree<BlockResumeTest.FakeState> build =
                new BlockTree<>(codec, CacheStore.inMemory(), 1 << 20, seed);
        long[] a = fp(15, 100);
        BlockResumeTest.FakeState s = new BlockResumeTest.FakeState();
        BlockTree<BlockResumeTest.FakeState>.Block tip = build.resume(new long[0], 0, s);
        s.ingestTo(12);
        tip = build.commit(tip, a, 0, 12, s);
        s.ingestTo(15);
        build.commit(tip, a, 12, 3, s);
        Path file = Files.createTempFile("frozen-corrupt", ".jkv");
        file.toFile().deleteOnExit();
        build.freeze(file);
        return file;
    }

    private static long indexOffset(byte[] artifact) {
        return ByteBuffer.wrap(artifact)
                .order(ByteOrder.LITTLE_ENDIAN)
                .getLong(FrozenBlocks.COUNT_OFFSET + 4);
    }

    /** The artifact with one little-endian int overwritten, in a fresh temp file. */
    private static byte[] mutateInt(byte[] pristine, long at, int value) {
        byte[] mutated = pristine.clone();
        ByteBuffer.wrap(mutated).order(ByteOrder.LITTLE_ENDIAN).putInt((int) at, value);
        return mutated;
    }

    private static byte[] mutateLong(byte[] pristine, long at, long value) {
        byte[] mutated = pristine.clone();
        ByteBuffer.wrap(mutated).order(ByteOrder.LITTLE_ENDIAN).putLong((int) at, value);
        return mutated;
    }

    private static void assertCorrupt(byte[] mutated, ContentKey seed) throws IOException {
        Path tmp = Files.createTempFile("frozen-mutated", ".jkv");
        tmp.toFile().deleteOnExit();
        Files.write(tmp, mutated);
        IllegalStateException e =
                assertThrows(IllegalStateException.class, () -> FrozenBlocks.open(tmp, seed));
        assertTrue(
                e.getMessage().contains("not a valid frozen prompt cache"),
                "the stable corrupt-artifact error, got: " + e.getMessage());
    }
}
