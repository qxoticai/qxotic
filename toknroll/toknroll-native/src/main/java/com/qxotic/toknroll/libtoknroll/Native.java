package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CLongPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.VoidPointer;
import org.graalvm.word.Pointer;
import org.graalvm.word.WordFactory;

/**
 * The toknroll__* entry points the C shim forwards to, one per header function. Every one catches
 * everything: an exception leaving a {@code @CEntryPoint} is fatal to the process, so a failure
 * becomes the thread's last error and a NULL or negative return. A handle is an {@link
 * ObjectHandles} word over a {@link Loaded}; a library that hosts these symbols next to its own
 * hands out handles with {@link #handle(Tokenizer)}.
 */
public final class Native {
    private static final ObjectHandles HANDLES = ObjectHandles.getGlobal();

    /** A tokenizer and its special tokens, built once per handle. */
    record Loaded(Tokenizer tokenizer, Specials specials) {
        static Loaded of(Tokenizer tokenizer) {
            return new Loaded(tokenizer, Specials.of(tokenizer));
        }
    }

    /** TOKNROLL_SPECIALS, the one option of toknroll_encode and toknroll_encode_batch. */
    private static final int SPECIALS = 1;

    /** Whether {@code flags} asks for specials; an unknown bit is an error. */
    private static boolean specials(int flags) {
        if ((flags & ~SPECIALS) != 0)
            throw Errors.invalid("unknown flags 0x" + Integer.toHexString(flags & ~SPECIALS));
        return flags == SPECIALS;
    }

    /** A C handle over a tokenizer another library already holds (libjinfer's models). */
    public static ObjectHandle handle(Tokenizer tokenizer) {
        return HANDLES.create(Loaded.of(tokenizer));
    }

    private static Loaded get(ObjectHandle handle) {
        if (handle.equal(WordFactory.zero())) throw Errors.invalid("tokenizer is NULL");
        Object value = HANDLES.get(handle);
        if (!(value instanceof Loaded loaded)) throw Errors.invalid("not a tokenizer handle");
        return loaded;
    }

    private static String string(CCharPointer value, String what) {
        if (value.isNull()) throw Errors.invalid(what + " is NULL");
        return CTypeConversion.toJavaString(value);
    }

    /** length bytes of UTF-8, any byte among them; NULL only with length 0. */
    private static String string(CCharPointer value, long length, String what) {
        if (length < 0) throw Errors.invalid("length must not be negative");
        if (length > Integer.MAX_VALUE) throw Errors.invalid(what + " is over 2 GiB");
        if (length == 0) return "";
        if (value.isNull()) throw Errors.invalid(what + " is NULL");
        return CTypeConversion.toJavaString(
                value, WordFactory.unsigned(length), StandardCharsets.UTF_8);
    }

    /** Room for a text's ids: prose runs about four bytes a token, so a third of its bytes. */
    private static int estimate(long bytes) {
        return (int) Math.min(Integer.MAX_VALUE - 8, Math.max(16, bytes / 3));
    }

    /**
     * {@code utf8[from, to)} encoded as the tokenizer encodes UTF-8, the specials matched first
     * when there are any: straight from the bytes when they are well-formed, else decoded first, as
     * every text was before, a bad sequence a U+FFFD.
     */
    static void encodeUtf8(
            Tokenizer tokenizer,
            Specials specials,
            byte[] utf8,
            int from,
            int to,
            IntSequence.Builder out) {
        if (WellFormed.utf8(utf8, from, to)) {
            if (specials == null) tokenizer.encodeUtf8Into(utf8, from, to, out);
            else specials.encodeUtf8Into(tokenizer, utf8, from, to, out);
        } else if (specials == null) tokenizer.encodeInto(decoded(utf8, from, to), out);
        else specials.encodeInto(tokenizer, decoded(utf8, from, to), out);
    }

    private static String decoded(byte[] utf8, int from, int to) {
        return new String(utf8, from, to - from, StandardCharsets.UTF_8);
    }

    /**
     * Bytes past a text copied in, so the engine reads its last 64-byte block whole: a masked
     * vector load, which the JIT compiles to one instruction, runs a lane at a time here.
     */
    private static final int SLACK = 64;

    /**
     * A calling thread's memory, kept for its next call so that a short one allocates nothing: the
     * text copied in, and the ids builder, read back through a live view. Either is let go once it
     * outgrows {@link #KEPT}.
     */
    private static final class Buffers {
        private static final int KEPT = 1 << 16;
        private byte[] text = new byte[0];
        private IntSequence.Builder ids;
        private IntSequence view; // live, over ids
        private byte[] chunk;

        /** KEPT bytes for a decode to go through. */
        byte[] chunk() {
            return chunk == null ? (chunk = new byte[KEPT]) : chunk;
        }

        byte[] text(int length) {
            if (text.length >= length + SLACK) return text;
            byte[] bigger = new byte[Math.max(length + SLACK, 2 * text.length)];
            if (bigger.length <= KEPT) text = bigger;
            return bigger;
        }

        /** The builder, empty. */
        IntSequence.Builder ids() {
            if (ids == null) {
                ids = IntSequence.newBuilder(256);
                view = ids.asSequenceView();
            } else {
                ImplAccessor.clear(ids);
            }
            return ids;
        }

        /** After a call: a builder past KEPT goes, so a long text leaves no array behind. */
        void trim() {
            if (ids.size() > KEPT) ids = null;
        }
    }

    private static final ThreadLocal<Buffers> BUFFERS = ThreadLocal.withInitial(Buffers::new);

    /**
     * From this many bytes or ids on, a copy across the boundary goes in bulk, through buffers:
     * their wrappers cost less than a loop does by then, and below it a loop allocates nothing.
     */
    private static final int BULK = 1 << 12;

    /** Ids written at a time in bulk: a few pages, so a long text costs no copy of itself. */
    private static final int WRITE_CHUNK = 1 << 14;

    /** {@code length} bytes of the caller's memory at {@code address} into {@code into}. */
    private static void copy(long address, byte[] into, int length) {
        if (length >= BULK) {
            CTypeConversion.asByteBuffer(WordFactory.pointer(address), length).get(into, 0, length);
            return;
        }
        Pointer from = WordFactory.pointer(address);
        int i = 0;
        // native order is little-endian on every platform the library is built for, as LONGS is
        for (; i + 8 <= length; i += 8) WellFormed.LONGS.set(into, i, from.readLong(i));
        for (; i < length; i++) into[i] = from.readByte(i);
    }

    /** {@code bytes[0, n)} into the caller's memory at {@code address}. */
    private static void write(byte[] bytes, long address, int n) {
        if (n >= BULK) {
            CTypeConversion.asByteBuffer(WordFactory.pointer(address), n).put(bytes, 0, n);
            return;
        }
        Pointer to = WordFactory.pointer(address);
        int i = 0;
        for (; i + 8 <= n; i += 8) to.writeLong(i, (long) WellFormed.LONGS.get(bytes, i));
        for (; i < n; i++) to.writeByte(i, bytes[i]);
    }

    /** {@code ids[from, from + n)} into the caller's memory at {@code address}. */
    private static void write(IntSequence ids, int from, long address, int n) {
        if (n >= BULK) {
            int[] chunk = new int[Math.min(n, WRITE_CHUNK)];
            for (int j = 0; j < n; j += chunk.length) {
                int m = Math.min(chunk.length, n - j);
                ids.copyTo(from + j, chunk, 0, m);
                CTypeConversion.asByteBuffer(WordFactory.pointer(address + 4L * j), 4 * m)
                        .order(ByteOrder.nativeOrder())
                        .asIntBuffer()
                        .put(chunk, 0, m);
            }
            return;
        }
        CIntPointer to = WordFactory.pointer(address);
        for (int i = 0; i < n; i++) to.write(i, ids.intAt(from + i));
    }

    /**
     * snprintf for bytes: writes min(n, byteCapacity) and a NUL, at byteCapacity - 1 when they do
     * not fit, unless byteCapacity is 0; returns n.
     */
    private static long bytes(byte[] value, CCharPointer out, long byteCapacity) {
        if (byteCapacity < 0) throw Errors.invalid("byte_capacity must not be negative");
        if (out.isNull() && byteCapacity > 0) throw Errors.invalid("out is NULL");
        int n = (int) Math.min(value.length, byteCapacity);
        if (n > 0) CTypeConversion.asByteBuffer(out, n).put(value, 0, n);
        if (byteCapacity > value.length) out.write(value.length, (byte) 0);
        else if (byteCapacity > 0)
            out.write((int) byteCapacity - 1, (byte) 0); // byteCapacity <= n here
        return value.length;
    }

    /** Tokens counted at a time, so no count can wrap: a token would need 512 KiB to do it. */
    private static final int COUNT_CHUNK = 1 << 12;

    /**
     * The bytes {@code tokens} decode to, refused past 2 GiB, where an int count would wrap.
     * Counted in chunks summed in 64 bits first; a chunk may count a byte short of what it decodes
     * to (Metaspace trims its first token's space), so the whole is counted once the sum fits, and
     * a total that overflows by that little wraps negative.
     */
    private static int countBytes(Tokenizer tokenizer, IntSequence tokens) {
        int n = tokens.length();
        if (n <= COUNT_CHUNK) return tokenizer.countBytes(tokens);
        long sum = 0;
        for (int i = 0; i < n; i += COUNT_CHUNK)
            sum += tokenizer.countBytes(tokens.subSequence(i, Math.min(n, i + COUNT_CHUNK)));
        int exact = sum > Integer.MAX_VALUE ? -1 : tokenizer.countBytes(tokens);
        if (exact < 0) throw Errors.invalid("the decoded text is over 2 GiB");
        return exact;
    }

    /** The shim's own failures (a configure call after the isolate exists) use the same channel. */
    @CEntryPoint(name = "toknroll__error_set")
    static void errorSet(IsolateThread thread, int code, CCharPointer message) {
        Errors.set(new Errors.Failure(code, CTypeConversion.toJavaString(message)));
    }

    @CEntryPoint(name = "toknroll__last_error")
    static CCharPointer lastError(IsolateThread thread, CIntPointer code) {
        if (code.isNonNull()) code.write(Errors.lastCode());
        return Errors.last();
    }

    /**
     * The isolate's first call, in libtoknroll and libjinfer alike: given is whether the host
     * called toknroll_log_set or jinfer_log_set, fn and userData what it passed.
     */
    @CEntryPoint(name = "toknroll__log_start")
    static void logStart(IsolateThread thread, int given, Log.Fn fn, VoidPointer userData) {
        try {
            Log.start(given != 0, fn, userData);
        } catch (Throwable t) { // ponytail: no caller to tell; the library runs, its lines lost
        }
    }

    @CEntryPoint(name = "toknroll__tokenizer_load")
    static ObjectHandle load(IsolateThread thread, CCharPointer source) {
        try {
            Errors.clear();
            return HANDLES.create(Loaded.of(Sources.load(string(source, "source"))));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "toknroll__tokenizer_free")
    static void free(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            get(handle);
            HANDLES.destroy(handle);
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__encode")
    static int encode(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointer text,
            long length,
            int flags,
            CIntPointer out,
            int tokenCapacity) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (length < 0) throw Errors.invalid("length must not be negative");
            if (length > Integer.MAX_VALUE) throw Errors.invalid("text is over 2 GiB");
            if (text.isNull() && length > 0) throw Errors.invalid("text is NULL");
            if (tokenCapacity < 0) throw Errors.invalid("token_capacity must not be negative");
            if (out.isNull() && tokenCapacity > 0) throw Errors.invalid("out is NULL");
            int n = (int) length;
            Buffers buffers = BUFFERS.get();
            byte[] utf8 = buffers.text(n);
            copy(text.rawValue(), utf8, n);
            IntSequence.Builder ids = buffers.ids();
            ids.ensureCapacity(estimate(n)); // a long text's ids in one array, not grown into
            Specials matcher = specials(flags) ? loaded.specials() : null;
            encodeUtf8(loaded.tokenizer(), matcher, utf8, 0, n, ids);
            int count = ids.size();
            write(buffers.view, 0, out.rawValue(), Math.min(count, tokenCapacity));
            buffers.trim();
            return count;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__encode_batch")
    static int encodeBatch(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointer texts,
            CLongPointer textOffsets,
            int count,
            int flags,
            CIntPointer out,
            int tokenCapacity,
            CIntPointer offsets) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (count < 0) throw Errors.invalid("count must not be negative");
            if (tokenCapacity < 0) throw Errors.invalid("token_capacity must not be negative");
            if (textOffsets.isNull() || offsets.isNull())
                throw Errors.invalid("text_offsets and offsets must not be NULL");
            if (out.isNull() && tokenCapacity > 0) throw Errors.invalid("out is NULL");
            long base = textOffsets.read(0);
            if (base < 0) throw Errors.invalid("text_offsets must not be negative");
            long[] starts = new long[count + 1]; // the texts' offsets from the first
            for (int i = 1; i <= count; i++) {
                starts[i] = textOffsets.read(i) - base;
                if (starts[i] < starts[i - 1])
                    throw Errors.invalid("text_offsets must not decrease, at " + i);
                if (starts[i] - starts[i - 1] > Integer.MAX_VALUE)
                    throw Errors.invalid("text " + (i - 1) + " is over 2^31 bytes");
            }
            if (texts.isNull() && starts[count] > 0) throw Errors.invalid("texts is NULL");
            Tokenizer tokenizer = loaded.tokenizer();
            Specials matcher = specials(flags) ? loaded.specials() : null;
            // raw addresses: a lambda in an image cannot capture a word
            long in = texts.rawValue() + base;
            // runs of texts of about equal bytes, several per core so a long text holds no core
            // back; a run copies its texts in through one buffer and keeps its ids in one array
            int[] runs = Runs.of(starts, count, 4 * Workers.THREADS);
            int[] counts = new int[count];
            IntSequence[] ids = new IntSequence[runs.length - 1];
            Workers.run(
                    ids.length,
                    r -> {
                        int first = runs[r];
                        int last = runs[r + 1];
                        int longest = 0;
                        for (int i = first; i < last; i++)
                            longest = Math.max(longest, (int) (starts[i + 1] - starts[i]));
                        // one buffer per run, small enough to stay in cache, each text
                        // copied in as it comes (one copy per run measured 6% slower)
                        byte[] text = new byte[longest + SLACK];
                        IntSequence.Builder b =
                                IntSequence.newBuilder(estimate(starts[last] - starts[first]));
                        for (int i = first; i < last; i++) {
                            int length = (int) (starts[i + 1] - starts[i]);
                            copy(in + starts[i], text, length);
                            int before = b.size();
                            encodeUtf8(tokenizer, matcher, text, 0, length, b);
                            counts[i] = b.size() - before;
                        }
                        ids[r] = b.build();
                    });
            int[] at = new int[count + 1]; // where text i's ids start
            offsets.write(0, 0);
            for (int i = 0; i < count; i++) {
                long next = (long) at[i] + counts[i];
                if (next > Integer.MAX_VALUE) throw Errors.invalid("the batch is over 2^31 ids");
                at[i + 1] = (int) next;
                offsets.write(i + 1, at[i + 1]);
            }
            // each run's ids written where they belong, on its own core
            long outAddress = out.rawValue();
            Workers.run(
                    ids.length,
                    r -> {
                        int from = at[runs[r]];
                        int n = Math.min(ids[r].length(), tokenCapacity - from);
                        write(ids[r], 0, outAddress + 4L * from, n);
                    });
            return at[count];
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__decode")
    static long decode(
            IsolateThread thread,
            ObjectHandle handle,
            CIntPointer ids,
            int count,
            int start,
            CCharPointer out,
            long byteCapacity,
            CIntPointer decodedTokens) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (count < 0) throw Errors.invalid("count must not be negative");
            if (ids.isNull() && count > 0) throw Errors.invalid("ids is NULL");
            if (start < 0 || start > count)
                throw Errors.invalid("start " + start + " is outside [0, count " + count + "]");
            if (byteCapacity < 0) throw Errors.invalid("byte_capacity must not be negative");
            if (out.isNull() && byteCapacity > 0) throw Errors.invalid("out is NULL");
            int n = count - start;
            if (n > Integer.MAX_VALUE / 4) throw Errors.invalid("ids are over 2 GiB");
            // ids[start, count) alone, behind a copy of its first at index 0 when start is past 0:
            // only a text's first token drops its leading space, and that copy's bytes are taken
            // back off the size
            int from = start > 0 && n > 0 ? 1 : 0; // where the ids begin in the copy
            int[] copy = new int[from + n];
            if (n > 0)
                CTypeConversion.asByteBuffer(ids.addressOf(start), 4 * n)
                        .order(ByteOrder.nativeOrder())
                        .asIntBuffer()
                        .get(copy, from, n);
            Tokenizer tokenizer = loaded.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            for (int i = from; i < copy.length; i++) {
                if (!vocab.contains(copy[i]))
                    throw Errors.invalid(
                            "token id "
                                    + copy[i]
                                    + " at index "
                                    + (start + i - from)
                                    + " is outside the vocabulary of "
                                    + vocab.size());
            }
            if (from == 1) copy[0] = copy[1];
            IntSequence tokens = IntSequence.wrap(copy);
            if (byteCapacity == 0) {
                long size = size(tokenizer, tokens, from); // refuses past 2 GiB
                if (decodedTokens.isNonNull()) decodedTokens.write(0);
                return size;
            }
            // the tokens' own bytes, whole tokens as far as they fit in byteCapacity - 1 (the NUL's
            // place): a chunk at a time through the thread's array, where the core stores words,
            // then out in bulk; a direct buffer over the caller's memory measured slower
            long room =
                    Math.min(byteCapacity - 1, Integer.MAX_VALUE); // past 2 GiB the count refuses
            byte[] chunk = BUFFERS.get().chunk();
            ByteBuffer buffer = ByteBuffer.wrap(chunk);
            int written = 0;
            int next = from; // the next id to decode
            while (next < copy.length) {
                buffer.clear().limit((int) Math.min(chunk.length, room - written));
                int consumed;
                try {
                    consumed = tokenizer.decodeBytesInto(tokens, next, buffer);
                } catch (IllegalArgumentException full) { // id next does not fit, the ids checked
                    if (room - written <= chunk.length) break; // nor would it in the room left
                    chunk = new byte[(int) Math.min(2L * chunk.length, room - written)]; // > 64 KB
                    buffer = ByteBuffer.wrap(chunk);
                    continue;
                }
                if (consumed <= 0) throw new IllegalStateException("decode made no progress");
                write(chunk, out.rawValue() + written, buffer.position());
                written += buffer.position();
                next += consumed;
            }
            // over capacity, the rest counted with what came before it: counted alone, its first
            // token would lose its space (Metaspace), and this is the slow pass anyway
            long size = next == copy.length ? written : size(tokenizer, tokens, from);
            out.write(written, (byte) 0);
            if (decodedTokens.isNonNull()) decodedTokens.write(next - from);
            return size;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    /**
     * The bytes tokens[from, n) take inside the whole: the whole less what tokens[0, from) take.
     */
    private static long size(Tokenizer tokenizer, IntSequence tokens, int from) {
        long whole = countBytes(tokenizer, tokens);
        return from == 0 ? whole : whole - tokenizer.countBytes(tokens.subSequence(0, from));
    }

    @CEntryPoint(name = "toknroll__vocabulary_size")
    static int vocabularySize(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle).tokenizer().vocabulary().size();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    /** TOKNROLL_TOKEN_*: bit n for GGUF type n, every type the vocabulary reports for the id. */
    @CEntryPoint(name = "toknroll__token_flags")
    static long tokenFlags(IsolateThread thread, ObjectHandle handle, int id) {
        try {
            Errors.clear();
            Vocabulary vocab = get(handle).tokenizer().vocabulary();
            if (!vocab.contains(id)) throw Errors.notFound("no token with id " + id);
            long flags = 0;
            for (StandardTokenType type : StandardTokenType.values()) {
                if (vocab.isTokenOfType(id, type)) flags |= 1L << type.getId();
            }
            return flags != 0 ? flags : 1L << StandardTokenType.NORMAL.getId();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__token")
    static long token(
            IsolateThread thread,
            ObjectHandle handle,
            int id,
            CCharPointer out,
            long byteCapacity) {
        try {
            Errors.clear();
            Vocabulary vocab = get(handle).tokenizer().vocabulary();
            if (!vocab.contains(id)) throw Errors.notFound("no token with id " + id);
            return bytes(vocab.token(id).getBytes(StandardCharsets.UTF_8), out, byteCapacity);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__id")
    static int id(IsolateThread thread, ObjectHandle handle, CCharPointer token, long length) {
        try {
            Errors.clear();
            Vocabulary vocab = get(handle).tokenizer().vocabulary();
            String s = string(token, length, "token");
            if (!vocab.contains(s)) throw Errors.notFound("no token '" + s + "'");
            return vocab.id(s);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    private Native() {}
}
