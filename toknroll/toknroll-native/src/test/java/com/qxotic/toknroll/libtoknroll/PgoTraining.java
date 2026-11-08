package com.qxotic.toknroll.libtoknroll;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.graalvm.nativeimage.CurrentIsolate;
import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.function.CEntryPointLiteral;
import org.graalvm.nativeimage.c.function.CFunctionPointer;
import org.graalvm.nativeimage.c.function.InvokeCFunctionPointer;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CLongPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.word.WordFactory;

/**
 * The workload the release build's profile is recorded on (`make libtoknroll` with Oracle GraalVM):
 * every header function, called as the C face calls it, through the image's own entry points
 * ({@link Native}) by function pointer, so the profile covers the boundary too, the copies in and
 * out, the buffers, the checks. Per tokenizer: toknroll_vocabulary_size, then toknroll_token_flags,
 * toknroll_token and toknroll_id over every id, which also find the special tokens; then
 * toknroll_encode and toknroll_encode_batch, plain and with specials, in 2 KB documents, on enwik8
 * prose, on chat text (a special token every 32-96 bytes), on FLORES-200 sentences in every script,
 * and on prose with a bad byte in every document (the encoder's ill-formed path); prose in 64 KB
 * documents too (the bulk copies, the chunked id write, the buffers let go); and toknroll_decode on
 * every document's ids, sized first (byte_capacity 0), every other one from start 1. Tokenizers of
 * every kind the library has a fast path for, and one from a GGUF.
 *
 * <p>Usage: PgoTraining <enwik8> <flores200> [source...], the {@link #SOURCES} when none is given.
 * Built with {@code --initialize-at-build-time} for this class: the entry point literals are
 * build-time constants.
 */
public final class PgoTraining {
    private PgoTraining() {}

    /**
     * A tokenizer per fast path: the engine's rules, core's cuts, its word encoder, SentencePiece;
     * fetched from the hub once. The last one is a GGUF, read for its header: the GGUF loader and
     * its vocabulary conversion.
     */
    static final String[] SOURCES = {
        "unsloth/Llama-3.2-1B-Instruct",
        "openai/gpt-oss-20b",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3.5-0.8B",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "LiquidAI/LFM3-1.7B",
        "google/gemma-4-e2b-it",
        "zai-org/GLM-5.1",
        "Qwen/Qwen3-0.6B-GGUF:Q8_0"
    };

    // The entry points, each as the C function pointer the image exports under toknroll__*.
    interface LoadFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        ObjectHandle invoke(IsolateThread thread, CCharPointer source);
    }

    interface FreeFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        void invoke(IsolateThread thread, ObjectHandle handle);
    }

    interface EncodeFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        int invoke(
                IsolateThread thread,
                ObjectHandle handle,
                CCharPointer text,
                long length,
                int flags,
                CIntPointer out,
                int tokenCapacity);
    }

    interface EncodeBatchFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        int invoke(
                IsolateThread thread,
                ObjectHandle handle,
                CCharPointer texts,
                CLongPointer textOffsets,
                int count,
                int flags,
                CIntPointer out,
                int tokenCapacity,
                CIntPointer offsets);
    }

    interface DecodeFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        long invoke(
                IsolateThread thread,
                ObjectHandle handle,
                CIntPointer ids,
                int count,
                int start,
                CCharPointer out,
                long byteCapacity,
                CIntPointer decodedTokens);
    }

    interface SizeFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        int invoke(IsolateThread thread, ObjectHandle handle);
    }

    interface FlagsFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        long invoke(IsolateThread thread, ObjectHandle handle, int id);
    }

    interface TokenFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        long invoke(
                IsolateThread thread, ObjectHandle handle, int id, CCharPointer out, long capacity);
    }

    interface IdFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        int invoke(IsolateThread thread, ObjectHandle handle, CCharPointer token, long length);
    }

    interface LastErrorFn extends CFunctionPointer {
        @InvokeCFunctionPointer
        CCharPointer invoke(IsolateThread thread, CIntPointer code);
    }

    private static final Class<?> H = ObjectHandle.class;
    private static final Class<?> T = IsolateThread.class;
    private static final CEntryPointLiteral<LoadFn> LOAD =
            CEntryPointLiteral.create(Native.class, "load", T, CCharPointer.class);
    private static final CEntryPointLiteral<FreeFn> FREE =
            CEntryPointLiteral.create(Native.class, "free", T, H);
    private static final CEntryPointLiteral<EncodeFn> ENCODE =
            CEntryPointLiteral.create(
                    Native.class,
                    "encode",
                    T,
                    H,
                    CCharPointer.class,
                    long.class,
                    int.class,
                    CIntPointer.class,
                    int.class);
    private static final CEntryPointLiteral<EncodeBatchFn> ENCODE_BATCH =
            CEntryPointLiteral.create(
                    Native.class,
                    "encodeBatch",
                    T,
                    H,
                    CCharPointer.class,
                    CLongPointer.class,
                    int.class,
                    int.class,
                    CIntPointer.class,
                    int.class,
                    CIntPointer.class);
    private static final CEntryPointLiteral<DecodeFn> DECODE =
            CEntryPointLiteral.create(
                    Native.class,
                    "decode",
                    T,
                    H,
                    CIntPointer.class,
                    int.class,
                    int.class,
                    CCharPointer.class,
                    long.class,
                    CIntPointer.class);
    private static final CEntryPointLiteral<SizeFn> VOCABULARY_SIZE =
            CEntryPointLiteral.create(Native.class, "vocabularySize", T, H);
    private static final CEntryPointLiteral<FlagsFn> TOKEN_FLAGS =
            CEntryPointLiteral.create(Native.class, "tokenFlags", T, H, int.class);
    private static final CEntryPointLiteral<TokenFn> TOKEN =
            CEntryPointLiteral.create(
                    Native.class, "token", T, H, int.class, CCharPointer.class, long.class);
    private static final CEntryPointLiteral<IdFn> ID =
            CEntryPointLiteral.create(Native.class, "id", T, H, CCharPointer.class, long.class);
    private static final CEntryPointLiteral<LastErrorFn> LAST_ERROR =
            CEntryPointLiteral.create(Native.class, "lastError", T, CIntPointer.class);

    /** TOKNROLL_SPECIALS. */
    private static final int SPECIALS = 1;

    /** TOKNROLL_TOKEN_CONTROL, USER_DEFINED and UNKNOWN: the tokens Specials matches. */
    private static final long SPECIAL_TYPES = 1L << 3 | 1L << 4 | 1L << 2;

    private static final int MB = 1 << 20;

    private static IsolateThread thread() {
        return CurrentIsolate.getCurrentThread();
    }

    private static RuntimeException failed(String call) {
        CCharPointer message =
                LAST_ERROR.getFunctionPointer().invoke(thread(), WordFactory.nullPointer());
        return new IllegalStateException(call + ": " + CTypeConversion.toJavaString(message));
    }

    public static void main(String[] args) throws Exception {
        String prose =
                new String(
                        Files.readAllBytes(Path.of(args[0])), 0, 12 * MB, StandardCharsets.UTF_8);
        byte[] multilingual = Files.readAllBytes(Path.of(args[1]));
        String[] sources = args.length > 2 ? Arrays.copyOfRange(args, 2, args.length) : SOURCES;
        for (int a = 0; a < sources.length; a++) {
            ObjectHandle tokenizer;
            try (CTypeConversion.CCharPointerHolder source =
                    CTypeConversion.toCString(sources[a])) {
                tokenizer = LOAD.getFunctionPointer().invoke(thread(), source.get());
            }
            if (tokenizer.equal(WordFactory.zero())) throw failed("toknroll_tokenizer_load");
            List<String> names = vocabulary(tokenizer);
            byte[] text = prose.substring(0, 8 * MB).getBytes(StandardCharsets.UTF_8);
            byte[] chat = chat(prose.substring(8 * MB), names, a + 1);
            byte[] bad = Arrays.copyOf(text, MB);
            for (int i = 512; i < bad.length; i += 1024) bad[i] = (byte) 0xFF;
            long ids = 0;
            for (byte[] t : new byte[][] {text, chat, multilingual, bad}) {
                for (int flags : new int[] {0, SPECIALS}) {
                    ids += documents(tokenizer, t, flags, false, 2048);
                    ids += documents(tokenizer, t, flags, true, 2048);
                }
            }
            ids += documents(tokenizer, text, 0, false, 64 << 10);
            ids += documents(tokenizer, text, 0, true, 64 << 10);
            FREE.getFunctionPointer().invoke(thread(), tokenizer);
            System.out.println(
                    sources[a] + ": " + names.size() + " specials, " + ids + " ids and bytes");
        }
    }

    /**
     * Every id's flags, spelling and the spelling's id back, as a C caller walks a vocabulary;
     * returns the special tokens' texts.
     */
    private static List<String> vocabulary(ObjectHandle tokenizer) {
        IsolateThread thread = thread();
        int size = VOCABULARY_SIZE.getFunctionPointer().invoke(thread, tokenizer);
        if (size < 0) throw failed("toknroll_vocabulary_size");
        List<String> names = new ArrayList<>();
        int capacity = 4096;
        CCharPointer out = UnmanagedMemory.malloc(WordFactory.unsigned(capacity));
        try {
            for (int id = 0; id < size; id++) {
                long flags = TOKEN_FLAGS.getFunctionPointer().invoke(thread, tokenizer, id);
                if (flags < 0) continue; // a gap in the vocabulary
                long n = TOKEN.getFunctionPointer().invoke(thread, tokenizer, id, out, capacity);
                if (n < 0) throw failed("toknroll_token");
                if (n >= capacity) continue; // ponytail: a token over 4 KB is skipped
                ID.getFunctionPointer().invoke(thread, tokenizer, out, n); // ponytail: not checked
                if ((flags & SPECIAL_TYPES) != 0)
                    names.add(
                            CTypeConversion.toJavaString(
                                    out, WordFactory.unsigned(n), StandardCharsets.UTF_8));
            }
        } finally {
            UnmanagedMemory.free(out);
        }
        return names;
    }

    /** Prose with one of the tokenizer's special tokens every 32 to 96 bytes. */
    private static byte[] chat(String prose, List<String> names, int seed) {
        if (names.isEmpty()) {
            return prose.getBytes(StandardCharsets.UTF_8);
        }
        Random random = new Random(seed);
        StringBuilder chat = new StringBuilder();
        for (int at = 0; at + 96 < prose.length(); ) {
            int step = 32 + random.nextInt(65);
            chat.append(prose, at, at + step).append(names.get(random.nextInt(names.size())));
            at += step;
        }
        return chat.toString().getBytes(StandardCharsets.UTF_8);
    }

    /**
     * {@code text} as documents of {@code docBytes}, cut at character boundaries, in the caller's
     * memory as a C caller holds it: each encoded on its own with toknroll_encode, or all of them
     * in one toknroll_encode_batch, and each one's ids sized and decoded; returns the ids and bytes
     * counted.
     */
    private static long documents(
            ObjectHandle tokenizer, byte[] text, int flags, boolean batch, int docBytes) {
        IsolateThread thread = thread();
        int count = (text.length + docBytes - 1) / docBytes;
        long[] starts = new long[count + 1];
        for (int i = 0; i < count; i++) starts[i] = start(text, i * docBytes);
        starts[count] = text.length;
        CCharPointer in = UnmanagedMemory.malloc(WordFactory.unsigned(text.length));
        CIntPointer ids =
                UnmanagedMemory.malloc(
                        WordFactory.unsigned(4L * text.length)); // an id covers a byte at least
        CIntPointer offsets = UnmanagedMemory.malloc(WordFactory.unsigned(4L * (count + 1)));
        CLongPointer textOffsets = UnmanagedMemory.malloc(WordFactory.unsigned(8L * (count + 1)));
        long decodeCapacity = 4L * docBytes + 64;
        CCharPointer decoded = UnmanagedMemory.malloc(WordFactory.unsigned(decodeCapacity));
        try {
            CTypeConversion.asByteBuffer(in, text.length).put(text);
            long total = 0;
            if (batch) {
                for (int i = 0; i <= count; i++) textOffsets.write(i, starts[i]);
                int n =
                        ENCODE_BATCH
                                .getFunctionPointer()
                                .invoke(
                                        thread,
                                        tokenizer,
                                        in,
                                        textOffsets,
                                        count,
                                        flags,
                                        ids,
                                        text.length,
                                        offsets);
                if (n < 0) throw failed("toknroll_encode_batch");
                total += n;
            } else {
                offsets.write(0, 0);
                for (int i = 0; i < count; i++) {
                    int at = offsets.read(i);
                    int n =
                            ENCODE.getFunctionPointer()
                                    .invoke(
                                            thread,
                                            tokenizer,
                                            in.addressOf(WordFactory.signed(starts[i])),
                                            starts[i + 1] - starts[i],
                                            flags,
                                            ids.addressOf(at),
                                            text.length - at);
                    if (n < 0) throw failed("toknroll_encode");
                    offsets.write(i + 1, at + n);
                    total += n;
                }
            }
            for (int i = 0; i < count; i++) {
                int at = offsets.read(i);
                int n = offsets.read(i + 1) - at;
                DecodeFn decode = DECODE.getFunctionPointer();
                long size =
                        decode.invoke(
                                thread,
                                tokenizer,
                                ids.addressOf(at),
                                n,
                                0,
                                WordFactory.nullPointer(),
                                0,
                                WordFactory.nullPointer());
                if (size < 0) throw failed("toknroll_decode (size)");
                long bytes =
                        decode.invoke(
                                thread,
                                tokenizer,
                                ids.addressOf(at),
                                n,
                                Math.min(i & 1, n),
                                decoded,
                                decodeCapacity,
                                WordFactory.nullPointer());
                if (bytes < 0) throw failed("toknroll_decode");
                total += bytes;
            }
            return total;
        } finally {
            UnmanagedMemory.free(decoded);
            UnmanagedMemory.free(textOffsets);
            UnmanagedMemory.free(offsets);
            UnmanagedMemory.free(ids);
            UnmanagedMemory.free(in);
        }
    }

    private static int start(byte[] text, int i) {
        while (i < text.length && (text[i] & 0xC0) == 0x80) {
            i++;
        }
        return i;
    }
}
