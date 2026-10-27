package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;
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

    /**
     * A tokenizer and its specials matcher, compiled once: every non-normal spelling, longest wins.
     */
    record Loaded(Tokenizer tokenizer, Specials specials) {
        static Loaded of(Tokenizer tokenizer) {
            Vocabulary vocab = tokenizer.vocabulary();
            Set<String> names = new HashSet<>();
            for (int id = 0; id < vocab.size(); id++) {
                if (vocab.contains(id) && !vocab.isTokenOfType(id, StandardTokenType.NORMAL))
                    names.add(vocab.token(id));
            }
            // a spelling that prefixes another is dropped: the matcher takes the longer one anyway
            Set<String> kept = new HashSet<>(names);
            for (String s : names) {
                for (String o : names) {
                    if (!s.equals(o) && o.startsWith(s)) {
                        kept.remove(s);
                        break;
                    }
                }
            }
            return new Loaded(
                    tokenizer, kept.isEmpty() ? Specials.none() : Specials.compile(vocab, kept));
        }
    }

    /**
     * The caller's output buffer as a builder: snprintf semantics, written up to capacity, counted
     * past it. Write-only during the call, so nothing of the caller's memory is retained. The
     * decode input is copied instead of viewed for the same reason: a view would be sound only as
     * long as no tokenizer keeps the sequence, which no API promises.
     */
    static final class NativeBuilder implements IntSequence.Builder {
        private final long address; // raw, as above
        private final int capacity;
        private int size;

        NativeBuilder(CIntPointer out, int capacity) {
            this.address = out.rawValue();
            this.capacity = capacity;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public void ensureCapacity(int minCapacity) {}

        @Override
        public IntSequence.Builder add(int value) {
            if (size < capacity) ((CIntPointer) WordFactory.pointer(address)).write(size, value);
            size++;
            return this;
        }

        @Override
        public IntSequence build() {
            // what was written, copied: the caller's buffer is theirs again once the call returns
            int[] written = new int[Math.min(size, capacity)];
            CIntPointer out = (CIntPointer) WordFactory.pointer(address);
            for (int i = 0; i < written.length; i++) written[i] = out.read(i);
            return IntSequence.wrap(written);
        }

        @Override
        public IntSequence snapshot() {
            return build();
        }

        @Override
        public IntSequence asSequenceView() {
            return build();
        }
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

    /** snprintf for bytes: writes min(n, capacity) and a NUL when there is room, returns n. */
    private static int bytes(byte[] value, CCharPointer out, int capacity) {
        if (capacity < 0) throw Errors.invalid("capacity must not be negative");
        if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
        int n = Math.min(value.length, capacity);
        for (int i = 0; i < n; i++) out.write(i, value[i]);
        if (capacity > value.length) out.write(value.length, (byte) 0);
        else if (capacity > 0) out.write(capacity - 1, (byte) 0);
        return value.length;
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

    @CEntryPoint(name = "toknroll__load")
    static ObjectHandle load(IsolateThread thread, CCharPointer source) {
        try {
            Errors.clear();
            return HANDLES.create(Loaded.of(Sources.load(string(source, "source"))));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "toknroll__free")
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
            boolean specials,
            CIntPointer out,
            int capacity) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            String s = string(text, length, "text");
            if (capacity < 0) throw Errors.invalid("capacity must not be negative");
            if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
            if (capacity == 0 && !specials) return loaded.tokenizer().countTokens(s);
            NativeBuilder builder =
                    new NativeBuilder(out, capacity); // ids land in the caller's buffer
            if (specials) loaded.specials().encodeInto(loaded.tokenizer(), s, builder);
            else loaded.tokenizer().encodeInto(s, builder);
            return builder.size();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__encode_batch")
    static int encodeBatch(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointer texts,
            CIntPointer textEnds,
            int count,
            boolean specials,
            CIntPointer out,
            int capacity,
            CIntPointer ends) {
        try {
            Errors.clear();
            Loaded loaded = get(handle);
            if (count < 0 || capacity < 0)
                throw Errors.invalid("count and capacity must not be negative");
            if ((texts.isNull() || textEnds.isNull() || ends.isNull()) && count > 0)
                throw Errors.invalid("texts, text_ends and ends must not be NULL");
            if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
            String[] strings = new String[count];
            for (int i = 0, start = 0; i < count; i++) {
                int end = textEnds.read(i);
                if (end < start) throw Errors.invalid("text_ends must not decrease, at " + i);
                strings[i] =
                        CTypeConversion.toJavaString(
                                texts.addressOf(start),
                                WordFactory.unsigned(end - start),
                                StandardCharsets.UTF_8);
                start = end;
            }
            Tokenizer tokenizer = loaded.tokenizer();
            // each text on its own core, then laid end to end in the caller's buffer
            int[][] ids =
                    IntStream.range(0, count)
                            .parallel()
                            .mapToObj(
                                    i ->
                                            specials
                                                    ? loaded.specials()
                                                            .encode(tokenizer, strings[i])
                                                            .toArray()
                                                    : tokenizer.encodeToArray(strings[i]))
                            .toArray(int[][]::new);
            int total = 0;
            for (int i = 0; i < count; i++) {
                for (int id : ids[i]) {
                    if (total < capacity) out.write(total, id);
                    total++;
                }
                ends.write(i, total);
            }
            return total;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "toknroll__decode")
    static int decode(
            IsolateThread thread,
            ObjectHandle handle,
            CIntPointer ids,
            int count,
            CCharPointer out,
            int capacity) {
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
            if (capacity == 0) return tokenizer.countBytes(tokens);
            // the tokens' own bytes, not a re-encoded string, so a partial UTF-8 sequence comes
            // back as it is (decode_bytes); they land in the caller's buffer, and only when they
            // do not fit are they sized and a prefix copied, as snprintf would
            ByteBuffer buffer = CTypeConversion.asByteBuffer(out, capacity - 1);
            try {
                decodeInto(tokenizer, tokens, buffer);
                out.write(buffer.position(), (byte) 0);
                return buffer.position();
            } catch (IllegalArgumentException full) {
                ByteBuffer whole = ByteBuffer.allocate(tokenizer.countBytes(tokens));
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
    static int token(
            IsolateThread thread, ObjectHandle handle, int id, CCharPointer out, int capacity) {
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
