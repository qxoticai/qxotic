package com.qxotic.jinfer.libjinfer;

import java.util.ArrayList;
import java.util.List;
import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CCharPointerPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.VoidPointer;
import org.graalvm.word.WordFactory;

/**
 * The jinfer__* entry points the C shim forwards to, one per header function. Every one catches
 * everything: an exception leaving a {@code @CEntryPoint} is fatal to the process, so a failure
 * becomes the thread's last error and a NULL or negative return. Handles are {@link ObjectHandles}
 * words; borrowed strings and buffers live on the handle they came from and go with it.
 */
public final class Native {
    private static final ObjectHandles HANDLES = ObjectHandles.getGlobal();

    private static <T> T get(ObjectHandle handle, Class<T> type, String what) {
        if (handle.equal(WordFactory.zero())) throw Errors.invalid(what + " is NULL");
        Object value = HANDLES.get(handle);
        if (!type.isInstance(value))
            throw Errors.invalid(
                    what + " is not a " + type.getSimpleName().toLowerCase() + " handle");
        return type.cast(value);
    }

    private static String string(CCharPointer value, String what) {
        if (value.isNull()) throw Errors.invalid(what + " is NULL");
        return CTypeConversion.toJavaString(value);
    }

    private static List<String> strings(CCharPointerPointer values, int count, String what) {
        if (count < 0) throw Errors.invalid("count must not be negative");
        if (values.isNull() && count > 0) throw Errors.invalid(what + " is NULL");
        List<String> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) out.add(string(values.read(i), what + "[" + i + "]"));
        return out;
    }

    /** snprintf for floats: writes min(n, capacity), returns n. */
    private static int floats(float[] values, CFloatPointer out, long capacity) {
        if (out.isNull() && capacity > 0) throw Errors.invalid("out is NULL");
        int n = Math.min(values.length, (int) Math.min(capacity, Integer.MAX_VALUE));
        for (int i = 0; i < n; i++) out.write(i, values[i]);
        return values.length;
    }

    /**
     * Frees the states a cancelled pass or a closed context abandoned. In a native image {@code
     * Arenas.newCrossThread()} is {@code Arena.ofAuto()} (GraalVM cannot close a shared arena), so
     * a state's {@code close()} frees nothing until a collection, and off-heap garbage never
     * provokes one: ~100 MB stayed resident per cancel. ponytail: a full collection of the small
     * Java heap costs milliseconds; a state pool in the engine would make it unnecessary.
     */
    static void collect() {
        System.gc();
    }

    // ---- library ----

    @CEntryPoint(name = "jinfer__last_error")
    static CCharPointer lastError(IsolateThread thread, CIntPointer code) {
        if (code.isNonNull()) code.write(Errors.lastCode());
        return Errors.last();
    }

    @CEntryPoint(name = "jinfer__log_set")
    static void logSet(IsolateThread thread, Log.Fn fn, VoidPointer payload) {
        try {
            Errors.clear();
            Log.set(fn, payload);
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    /** The shim's own failures (a configure call after the isolate exists) use the same channel. */
    @CEntryPoint(name = "jinfer__error_set")
    static void errorSet(IsolateThread thread, int code, CCharPointer message) {
        Errors.set(new Errors.Failure(code, CTypeConversion.toJavaString(message)));
    }

    // ---- model ----

    @CEntryPoint(name = "jinfer__model_load")
    static ObjectHandle modelLoad(IsolateThread thread, CCharPointer ref, Options.Struct options) {
        try {
            Errors.clear();
            return HANDLES.create(Model.load(string(ref, "ref"), Options.read(options)));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__model_free")
    static void modelFree(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            Model model = get(handle, Model.class, "model");
            HANDLES.destroy(handle);
            model.release();
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__model_property")
    static int modelProperty(IsolateThread thread, ObjectHandle handle, int key) {
        try {
            Errors.clear();
            return get(handle, Model.class, "model").property(key);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__model_info")
    static CCharPointer modelInfo(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle, Model.class, "model").info().get();
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__model_tokenizer")
    static ObjectHandle modelTokenizer(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            // a toknroll handle over the model's tokenizer, freed by toknroll_free; the tokenizer
            // is a plain heap object, so it outlives the model
            return com.qxotic.toknroll.libtoknroll.Native.handle(
                    get(handle, Model.class, "model").tokenizer());
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    // ---- state ----

    @CEntryPoint(name = "jinfer__state_new")
    static ObjectHandle stateNew(IsolateThread thread, ObjectHandle handle, int contextCapacity) {
        try {
            Errors.clear();
            return HANDLES.create(new State(get(handle, Model.class, "model"), contextCapacity));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__state_free")
    static void stateFree(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            State state = get(handle, State.class, "state");
            HANDLES.destroy(handle);
            state.close();
            state = null; // garbage only once nothing on this stack holds it
            collect();
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__state_reset")
    static int stateReset(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            get(handle, State.class, "state").reset();
            return Errors.OK;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__state_position")
    static int statePosition(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle, State.class, "state").position();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__state_ingest")
    static int ingest(IsolateThread thread, ObjectHandle handle, CIntPointer tokens, int count) {
        try {
            Errors.clear();
            State state = get(handle, State.class, "state");
            if (count < 0) throw Errors.invalid("count must not be negative");
            if (tokens.isNull() && count > 0) throw Errors.invalid("tokens is NULL");
            int[] ids = new int[count];
            for (int i = 0; i < count; i++) ids[i] = tokens.read(i);
            return state.ingest(ids);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__state_logits")
    static int logits(IsolateThread thread, ObjectHandle handle, CFloatPointer out, int capacity) {
        try {
            Errors.clear();
            return floats(get(handle, State.class, "state").logits(), out, capacity);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__state_embedding")
    static int embedding(
            IsolateThread thread, ObjectHandle handle, CFloatPointer out, int capacity) {
        try {
            Errors.clear();
            return floats(get(handle, State.class, "state").embedding(), out, capacity);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    // ---- stream ----

    @CEntryPoint(name = "jinfer__stream_step")
    static int streamStep(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle, Stream.class, "stream").step();
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__stream_text")
    static CCharPointer streamText(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle, Stream.class, "stream").cText();
        } catch (Throwable t) {
            Errors.set(t);
            return Errors.last(); // never NULL for text, as the header says
        }
    }

    @CEntryPoint(name = "jinfer__stream_samples")
    static CFloatPointer streamSamples(
            IsolateThread thread, ObjectHandle handle, CIntPointer count) {
        try {
            Errors.clear();
            Stream stream = get(handle, Stream.class, "stream");
            float[] pcm = stream.samples;
            if (count.isNonNull()) count.write(pcm == null ? 0 : pcm.length);
            return stream.cSamples();
        } catch (Throwable t) {
            Errors.set(t);
            if (count.isNonNull()) count.write(0);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__stream_result")
    static CCharPointer streamResult(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            return get(handle, Stream.class, "stream").cResult();
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__stream_cancel")
    static void streamCancel(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            get(handle, Stream.class, "stream").cancel();
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__stream_free")
    static void streamFree(IsolateThread thread, ObjectHandle handle) {
        try {
            Errors.clear();
            Stream stream = get(handle, Stream.class, "stream");
            HANDLES.destroy(handle);
            stream.close();
        } catch (Throwable t) {
            Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__chat")
    static ObjectHandle chat(IsolateThread thread, ObjectHandle handle, CCharPointer requestJson) {
        try {
            Errors.clear();
            return HANDLES.create(
                    Stream.chat(
                            get(handle, Model.class, "model"),
                            string(requestJson, "request_json")));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__speak")
    static ObjectHandle speak(IsolateThread thread, ObjectHandle handle, CCharPointer requestJson) {
        try {
            Errors.clear();
            return HANDLES.create(
                    Stream.speech(
                            get(handle, Model.class, "model"),
                            string(requestJson, "request_json")));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    @CEntryPoint(name = "jinfer__transcribe")
    static ObjectHandle transcribe(
            IsolateThread thread,
            ObjectHandle handle,
            CFloatPointer pcm,
            int count,
            int sampleRate) {
        try {
            Errors.clear();
            Model model = get(handle, Model.class, "model");
            if (count < 0) throw Errors.invalid("count must not be negative");
            if (pcm.isNull() && count > 0) throw Errors.invalid("pcm is NULL");
            float[] samples = new float[count];
            for (int i = 0; i < samples.length; i++) samples[i] = pcm.read(i);
            return HANDLES.create(Stream.transcription(model, samples, sampleRate));
        } catch (Throwable t) {
            Errors.set(t);
            return WordFactory.nullPointer();
        }
    }

    // ---- retrieval, prompt cache ----

    @CEntryPoint(name = "jinfer__embed")
    static int embed(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointerPointer texts,
            int count,
            int dimension,
            CFloatPointer out,
            int capacity) {
        try {
            Errors.clear();
            Model model = get(handle, Model.class, "model");
            if (capacity < 0) throw Errors.invalid("capacity must not be negative");
            float[] all = model.embed(strings(texts, count, "texts"), dimension);
            return floats(all, out, capacity);
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__rerank")
    static int rerank(
            IsolateThread thread,
            ObjectHandle handle,
            CCharPointer query,
            CCharPointerPointer documents,
            int count,
            CCharPointer instruction,
            CFloatPointer scores) {
        try {
            Errors.clear();
            Model model = get(handle, Model.class, "model");
            String q = string(query, "query");
            List<String> docs = strings(documents, count, "documents");
            if (scores.isNull() && count > 0) throw Errors.invalid("scores is NULL");
            String task = instruction.isNull() ? null : CTypeConversion.toJavaString(instruction);
            floats(model.rerank(q, docs, task), scores, count);
            return Errors.OK;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    @CEntryPoint(name = "jinfer__prepare")
    static int prepare(IsolateThread thread, ObjectHandle handle, CCharPointer requestJson) {
        try {
            Errors.clear();
            Stream.prepare(get(handle, Model.class, "model"), string(requestJson, "request_json"));
            return Errors.OK;
        } catch (Throwable t) {
            return Errors.set(t);
        }
    }

    private Native() {}
}
