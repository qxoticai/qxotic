package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HexFormat;

/** A compiled single prompt: the resume-state of one fixed token prefix, sealed into one file at
 *  build time and restored in milliseconds at serve time — the "prompt compiler" artifact for
 *  CPU deployments (native image + sidecar file). Tree-less and hash-less on purpose: one known
 *  prompt needs no keys or matching, just a fingerprint memcmp. A mismatch (different prompt,
 *  different model) is always a clean fall-through to plain prefill, never a wrong restore.
 *
 *  <p>Layout (little-endian): {@code JKVS, formatVersion, modelSeed[32], nameLen+nameUtf8,
 *  fingerprintCount N, pad to 64} then {@code N} fingerprint longs, pad to 64, then the KV span
 *  ({@link StateCodec#save} of {@code [0,N)}). {@link #open} maps the file lazily — header and
 *  fingerprints are the only pages touched before a restore. The model seed (see
 *  {@link PromptCache#modelSeed}) also covers the codec blob layout: any layout change ships as a
 *  seed/format bump, so a stale file fails validation instead of restoring garbage. */
public final class SealedPrompt {

    private static final int MAGIC = 0x53564B4A;          // "JKVS"
    private static final int FORMAT_VERSION = 2;      // v2: [rows][checkpoint] span layout
    private static final int ALIGN = 64;

    private final Path file;
    private final String name;
    private final MemorySegment fingerprints;             // N longs, mapped
    private final MemorySegment kv;                       // the sealed span, mapped (lazy)

    private SealedPrompt(Path file, String name, MemorySegment fingerprints, MemorySegment kv) {
        this.file = file;
        this.name = name;
        this.fingerprints = fingerprints;
        this.kv = kv;
    }

    /** Seals {@code state} — which must hold exactly the {@code fingerprints.length} positions of
     *  the compiled prompt — into {@code out}. */
    public static <S extends RuntimeState> void compile(
            Path out, String name, StateCodec<S> codec, S state, long[] fingerprints, byte[] modelSeed) throws IOException {
        int n = fingerprints.length;
        if (state.position() != n) {
            throw new IllegalStateException("state at " + state.position() + ", prompt is " + n + " positions");
        }
        byte[] nameUtf8 = name.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        long fpOffset = align(4 + 4 + 32 + 4 + nameUtf8.length + 4);
        long kvOffset = align(fpOffset + (long) n * Long.BYTES);
        long kvBytes = codec.rowBytes(n) + codec.checkpointBytes();
        try (FileChannel ch = FileChannel.open(out, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.READ, StandardOpenOption.WRITE);
             Arena arena = Arena.ofConfined()) {
            MemorySegment map = ch.map(FileChannel.MapMode.READ_WRITE, 0, kvOffset + kvBytes, arena);
            ByteBuffer header = map.asSlice(0, fpOffset).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(MAGIC).putInt(FORMAT_VERSION).put(modelSeed, 0, 32)
                    .putInt(nameUtf8.length).put(nameUtf8).putInt(n);
            MemorySegment fp = map.asSlice(fpOffset, (long) n * Long.BYTES);
            for (int i = 0; i < n; i++) fp.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, i, fingerprints[i]);
            MemorySegment kv = map.asSlice(kvOffset, kvBytes);
            codec.saveRows(state, 0, n, kv);
            codec.saveCheckpoint(state, n, kv.asSlice(codec.rowBytes(n)));   // sealed span: always checkpointed
            map.force();
        }
    }

    /** Maps {@code file} lazily and validates it belongs to the model identified by
     *  {@code modelSeed}; throws a descriptive error when it does not. The returned instance holds
     *  the mapping for the JVM's lifetime (global arena) — sealed prompts are process-lifetime
     *  artifacts. */
    public static SealedPrompt open(Path file, byte[] modelSeed) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            MemorySegment map = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), Arena.global());
            ByteBuffer header = map.asSlice(0, Math.min(map.byteSize(), 4096)).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            if (header.getInt() != MAGIC) {
                throw new IllegalStateException(file + " is not a sealed prompt (bad magic)");
            }
            int version = header.getInt();
            if (version != FORMAT_VERSION) {
                throw new IllegalStateException(file + " has sealed-prompt format v" + version
                        + ", this build reads v" + FORMAT_VERSION + "; recompile the prompt");
            }
            byte[] seed = new byte[32];
            header.get(seed);
            byte[] nameUtf8 = new byte[header.getInt()];
            header.get(nameUtf8);
            String name = new String(nameUtf8, java.nio.charset.StandardCharsets.UTF_8);
            if (!java.util.Arrays.equals(seed, 0, 32, modelSeed, 0, 32)) {
                throw new IllegalStateException("sealed prompt " + file + " ('" + name
                        + "', model seed " + hex(seed) + ") was built for a different model than the one loaded (seed "
                        + hex(modelSeed) + "); the cache is model-specific - recompile it or load the matching GGUF");
            }
            int n = header.getInt();
            long fpOffset = align(4 + 4 + 32 + 4 + nameUtf8.length + 4);
            long kvOffset = align(fpOffset + (long) n * Long.BYTES);
            return new SealedPrompt(file, name,
                    map.asSlice(fpOffset, (long) n * Long.BYTES),
                    map.asSlice(kvOffset, map.byteSize() - kvOffset));
        }
    }

    /** Positions the sealed prompt covers. */
    public int positions() {
        return (int) (fingerprints.byteSize() / Long.BYTES);
    }

    public String name() {
        return name;
    }

    /** Restores the sealed prompt into {@code state} iff it is a prefix of {@code requestFp}
     *  (plain comparison, no hashing) and returns the positions resumed; returns 0 on any mismatch
     *  — the caller prefills normally, a miss never degrades correctness. */
    public <S extends RuntimeState> int tryRestore(S state, StateCodec<S> codec, long[] requestFp) {
        int n = positions();
        if (requestFp.length < n) return 0;
        for (int i = 0; i < n; i++) {
            if (fingerprints.getAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED, i) != requestFp[i]) return 0;
        }
        codec.restoreRows(state, 0, n, kv);
        codec.restoreCheckpoint(state, n, kv.asSlice(codec.rowBytes(n)));
        state.resumeAt(n);
        return n;
    }

    private static long align(long offset) {
        return (offset + ALIGN - 1) & -ALIGN;
    }

    private static String hex(byte[] b) {
        return HexFormat.of().formatHex(b, 0, 8) + "...";
    }

    @Override
    public String toString() {
        return "SealedPrompt[" + name + ", " + positions() + " positions, " + file + "]";
    }
}
