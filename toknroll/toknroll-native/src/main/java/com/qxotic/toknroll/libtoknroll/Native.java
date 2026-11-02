package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.word.UnsignedWord;
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
    private static String string(CCharPointer value, UnsignedWord length, String what) {
        if (length.aboveThan(Integer.MAX_VALUE)) throw Errors.invalid(what + " is over 2 GiB");
        if (length.equal(0)) return "";
        if (value.isNull()) throw Errors.invalid(what + " is NULL");
        return CTypeConversion.toJavaString(value, length, StandardCharsets.UTF_8);
    }

    /** Room for a text's ids: prose runs about four bytes a token, so a third of its bytes. */
    private static int estimate(int bytes) {
        return Math.max(16, bytes / 3);
    }

    /**
     * {@code utf8[from, to)} encoded as the tokenizer encodes UTF-8, the specials matched first
     * when there are any: straight from the bytes when they are well-formed, else decoded first, as
     * every text was before, a bad sequence a U+FFFD.
     */
    private static void encodeUtf8(
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

    /** {@code length} bytes of the caller's memory as a Java array: one copy, read once. */
    private static byte[] copy(CCharPointer value, int length) {
        byte[] bytes = new byte[length];
        copy(value.rawValue(), bytes, length);
        return bytes;
    }

    private static void copy(long address, byte[] into, int length) {
        if (length > 0)
            CTypeConversion.asByteBuffer(WordFactory.pointer(address), length).get(into, 0, length);
    }

    /**
     * Ids moved to the caller's memory at a time: a few pages, so a long text costs no copy of
     * itself.
     */
    private static final int WRITE_CHUNK = 1 << 14;

    /** The first {@code n} of {@code ids} into the caller's memory at {@code address}, in bulk. */
    private static void write(IntSequence ids, long address, int n) {
        int[] chunk = new int[Math.min(n, WRITE_CHUNK)];
        for (int j = 0; j < n; j += chunk.length) {
            int m = Math.min(chunk.length, n - j);
            ids.copyTo(j, chunk, 0, m);
            CTypeConversion.asByteBuffer(WordFactory.pointer(address + 4L * j), 4 * m)
                    .order(ByteOrder.nativeOrder())
                    .asIntBuffer()
                    .put(chunk, 0, m);
        }
    }

    /**
     * snprintf for bytes: writes min(n, capacity) and a NUL, at capacity - 1 when they do not fit,
     * unless capacity is 0; returns n.
     */
    private static long bytes(byte[] value, CCharPointer out, long capacity) {
        if (capacity < 0) throw Errors.invalid("capacity must not be negative");
        if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
        int n = (int) Math.min(value.length, capacity);
        for (int i = 0; i < n; i++) out.write(i, value[i]);
        if (capacity > value.length) out.write(value.length, (byte) 0);
        else if (capacity > 0) out.write((int) capacity - 1, (byte) 0); // capacity <= n here
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
            UnsignedWord length,
            int flags,
            CIntPointer out,
            int capacity) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (length.aboveThan(Integer.MAX_VALUE)) throw Errors.invalid("text is over 2 GiB");
            if (text.isNull() && length.aboveThan(0)) throw Errors.invalid("text is NULL");
            if (capacity < 0) throw Errors.invalid("capacity must not be negative");
            if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
            int n = (int) length.rawValue();
            byte[] utf8 = copy(text, n);
            IntSequence.Builder ids = IntSequence.newBuilder(estimate(n));
            Specials matcher = specials(flags) ? loaded.specials() : null;
            encodeUtf8(loaded.tokenizer(), matcher, utf8, 0, n, ids);
            IntSequence encoded = ids.build();
            write(encoded, out.rawValue(), Math.min(encoded.length(), capacity));
            return encoded.length();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__encode_batch")
    static int encodeBatch(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointer texts,
            CIntPointer textOffsets,
            int count,
            int flags,
            CIntPointer out,
            int capacity,
            CIntPointer offsets) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (count < 0 || capacity < 0)
                throw Errors.invalid("count and capacity must not be negative");
            if (textOffsets.isNull() || offsets.isNull())
                throw Errors.invalid("text_offsets and offsets must not be NULL");
            if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
            int base = textOffsets.read(0);
            if (base < 0) throw Errors.invalid("text_offsets must not be negative");
            int[] starts = new int[count + 1]; // the texts' offsets from the first
            for (int i = 1; i <= count; i++) {
                starts[i] = textOffsets.read(i) - base;
                if (starts[i] < starts[i - 1])
                    throw Errors.invalid("text_offsets must not decrease, at " + i);
            }
            if (texts.isNull() && starts[count] > 0) throw Errors.invalid("texts is NULL");
            Tokenizer tokenizer = loaded.tokenizer();
            Specials matcher = specials(flags) ? loaded.specials() : null;
            // raw addresses: a lambda in an image cannot capture a word
            long in = texts.rawValue() + base;
            // runs of texts of about equal bytes, several per core so a long text holds no core
            // back; a run copies its texts in through one buffer and keeps its ids in one array
            int[] runs = Runs.of(starts, count, 4 * (ForkJoinPool.getCommonPoolParallelism() + 1));
            int[] counts = new int[count];
            IntSequence[] ids = new IntSequence[runs.length - 1];
            IntStream.range(0, ids.length)
                    .parallel()
                    .forEach(
                            r -> {
                                int first = runs[r];
                                int last = runs[r + 1];
                                int longest = 0;
                                for (int i = first; i < last; i++)
                                    longest = Math.max(longest, starts[i + 1] - starts[i]);
                                // one buffer per run, small enough to stay in cache, each text
                                // copied in as it comes (one copy per run measured 6% slower)
                                byte[] text = new byte[longest];
                                IntSequence.Builder b =
                                        IntSequence.newBuilder(
                                                estimate(starts[last] - starts[first]));
                                for (int i = first; i < last; i++) {
                                    int length = starts[i + 1] - starts[i];
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
            IntStream.range(0, ids.length)
                    .parallel()
                    .forEach(
                            r -> {
                                int from = at[runs[r]];
                                int n = Math.min(ids[r].length(), capacity - from);
                                if (n > 0) write(ids[r], outAddress + 4L * from, n);
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
            CCharPointer out,
            long capacity) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (count < 0) throw Errors.invalid("count must not be negative");
            if (ids.isNull() && count > 0) throw Errors.invalid("ids is NULL");
            if (capacity < 0) throw Errors.invalid("capacity must not be negative");
            if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
            Vocabulary vocab = loaded.tokenizer().vocabulary();
            int[] copy = new int[count];
            for (int i = 0; i < count; i++) {
                copy[i] = ids.read(i);
                if (!vocab.contains(copy[i]))
                    throw Errors.invalid(
                            "token id "
                                    + copy[i]
                                    + " at index "
                                    + i
                                    + " is outside the vocabulary of "
                                    + vocab.size());
            }
            IntSequence tokens = IntSequence.wrap(copy);
            Tokenizer tokenizer = loaded.tokenizer();
            if (capacity == 0) return countBytes(tokenizer, tokens);
            // the tokens' own bytes, not a re-encoded string, so a partial UTF-8 sequence comes
            // back as it is (decode_bytes); they land in the caller's buffer, and only when they
            // do not fit are they sized and a prefix copied, as snprintf would
            int room = (int) Math.min(capacity - 1, Integer.MAX_VALUE);
            ByteBuffer buffer = CTypeConversion.asByteBuffer(out, room);
            try {
                decodeInto(tokenizer, tokens, buffer);
                out.write(buffer.position(), (byte) 0);
                return buffer.position();
            } catch (IllegalArgumentException full) {
                ByteBuffer whole = ByteBuffer.allocate(countBytes(tokenizer, tokens));
                decodeInto(tokenizer, tokens, whole);
                return bytes(whole.array(), out, capacity);
            }
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    private static void decodeInto(Tokenizer tokenizer, IntSequence tokens, ByteBuffer buffer) {
        for (int index = 0; index < tokens.length(); ) {
            int consumed = tokenizer.decodeBytesInto(tokens, index, buffer);
            if (consumed <= 0) throw new IllegalStateException("decode made no progress");
            index += consumed;
        }
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
            IsolateThread thread, ObjectHandle handle, int id, CCharPointer out, long capacity) {
        try {
            Errors.clear();
            Vocabulary vocab = get(handle).tokenizer().vocabulary();
            if (!vocab.contains(id)) throw Errors.notFound("no token with id " + id);
            return bytes(vocab.token(id).getBytes(StandardCharsets.UTF_8), out, capacity);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__id")
    static int id(
            IsolateThread thread, ObjectHandle handle, CCharPointer token, UnsignedWord length) {
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
