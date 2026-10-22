package com.qxotic.jinfer.libjinfer;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.server.OpenAiChat;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;
import org.graalvm.word.WordFactory;

/**
 * Every generation, stepped like a statement: a producer thread pushes chunks into a queue, the
 * caller's thread takes one per step. A chunk is text or audio. Freeing a running stream cancels it
 * and waits for the producer to let go of the model. The C copies of the current chunk and the
 * result are borrowed from the stream, as the header says, and freed with it.
 */
final class Stream {
    private static final Object END = new Object();

    private final Model model;
    private final LinkedBlockingQueue<Object> chunks = new LinkedBlockingQueue<>();
    private volatile boolean cancelled;
    private volatile boolean finished;
    String text = "";
    float[] samples;
    String result;
    private CCharPointerHolder cText, cResult;
    private long cSamples; // raw: a word may not live in a field

    private Stream(Model model) {
        this.model = model;
        model.retain();
    }

    private void produce(Runnable body, Executor executor) {
        try {
            executor.execute(
                    () -> {
                        try {
                            body.run();
                            chunks.add(END);
                        } catch (Throwable t) {
                            chunks.add(t);
                        }
                    });
        } catch (RuntimeException rejected) {
            model.release();
            throw rejected;
        }
    }

    // ---- chat: the OpenAI chat.completions dialect is jinfer-server's own ----

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parse(String requestJson) {
        Object parsed;
        try {
            parsed = nulls(Json.parse(requestJson));
        } catch (Json.ParseException e) {
            throw Errors.invalid("request_json is not valid JSON: " + e.getMessage());
        }
        if (parsed instanceof Map<?, ?> map) return (Map<String, Object>) map;
        throw Errors.invalid("request_json must be a JSON object");
    }

    /**
     * Json.NULL to Java null, in place, as the server's JsonCodec reads a request: an assistant
     * tool-call message has {@code "content": null}. ponytail: the server's codec is
     * package-private; a public {@code OpenAiChat.lower(String json, ...)} would make this one
     * parse, in one place.
     */
    @SuppressWarnings("unchecked")
    private static Object nulls(Object value) {
        if (value == Json.NULL) return null;
        if (value instanceof Map<?, ?> map)
            for (var entry : ((Map<String, Object>) map).entrySet())
                entry.setValue(nulls(entry.getValue()));
        else if (value instanceof List<?> list) ((List<Object>) list).replaceAll(Stream::nulls);
        return value;
    }

    private static OpenAiChat.Lowered lower(Model model, Map<String, Object> wire) {
        ChatEngine engine = model.engine();
        return OpenAiChat.lower(
                wire, model.name, Model.CONFIG, engine.loaded().samplingDefaults().resolve());
    }

    /** jinfer_prepare: the request's prompt computed into the cache now, nothing generated. */
    static void prepare(Model model, String requestJson) {
        model.engine().definePrompt(lower(model, parse(requestJson)).request());
    }

    static Stream chat(Model model, String requestJson) {
        Map<String, Object> wire = parse(requestJson);
        OpenAiChat.Lowered lowered = lower(model, wire);
        ChatEngine engine = model.engine();
        String id = OpenAiChat.newId();
        String modelId = OpenAiChat.modelId(wire, model.name);
        Stream s = new Stream(model);
        StringBuilder produced =
                new StringBuilder(); // the words so far: a cancelled pass keeps them
        ChatEngine.ReplySink sink =
                new ChatEngine.ReplySink() {
                    @Override
                    public void on(ChatEngine.Delta delta) {
                        if (delta.channel() != Channel.CONTENT) return;
                        s.chunks.add(delta.text());
                        produced.append(delta.text());
                    }

                    @Override
                    public boolean cancelled() {
                        return s.cancelled;
                    }
                };
        s.produce(
                () -> {
                    ChatEngine.Completion completion = engine.complete(lowered.request(), sink);
                    Message said =
                            produced.isEmpty() ? null : Message.assistant(produced.toString());
                    s.result = OpenAiChat.completion(id, modelId, completion, lowered, said);
                },
                engine::stream);
        return s;
    }

    /** jinfer_speak: the /v1/audio/speech fields that apply, {@code input} and {@code speed}. */
    static Stream speech(Model model, String requestJson) {
        if (model.speech == null)
            throw Errors.unsupported(
                    model.name + " is " + model.article() + ", not a speech model");
        Map<String, Object> wire = parse(requestJson);
        if (!(wire.get("input") instanceof String text) || text.isEmpty())
            throw Errors.invalid("request_json needs a non-empty \"input\" string");
        Object speedValue = wire.get("speed");
        if (speedValue != null && !(speedValue instanceof Number))
            throw Errors.invalid("\"speed\" must be a number");
        double speed = speedValue == null ? 0 : ((Number) speedValue).doubleValue();
        if (speedValue != null && speed <= 0) throw Errors.invalid("\"speed\" must be positive");
        Stream s = new Stream(model);
        SpeechOptions options = speed > 0 ? SpeechOptions.speed(speed) : SpeechOptions.NONE;
        s.produce(
                () -> {
                    long[] total = {0};
                    model.speech.speak(
                            text,
                            options,
                            audio -> {
                                total[0] += audio.pcm().length;
                                s.chunks.add(audio.pcm());
                                return !s.cancelled;
                            });
                    Map<String, Object> out = new LinkedHashMap<>();
                    out.put("sample_rate", model.speech.sampleRate());
                    out.put("samples", total[0]);
                    out.put("seconds", total[0] / (double) model.speech.sampleRate());
                    s.result = Json.stringify(out);
                },
                model.driver());
        return s;
    }

    static Stream transcription(Model model, float[] pcm, int sampleRate) {
        if (model.transcription == null)
            throw Errors.unsupported(
                    model.name + " is " + model.article() + ", not a transcription model");
        if (sampleRate <= 0) throw Errors.invalid("sample_rate must be positive");
        Stream s = new Stream(model);
        s.produce(
                () -> {
                    int target = model.transcription.sampleRate();
                    float[] samples = Resample.to(pcm, sampleRate, target);
                    Transcription t =
                            model.transcription.transcribe(new Media.Audio(samples, target, 1));
                    if (!t.text().isEmpty()) s.chunks.add(t.text());
                    List<Object> words = new ArrayList<>();
                    for (Transcription.Word w : t.words()) {
                        Map<String, Object> word = new LinkedHashMap<>();
                        word.put("text", w.text());
                        word.put("start", w.start().toNanos() / 1e9);
                        word.put("end", w.end().toNanos() / 1e9);
                        word.put("confidence", w.confidence());
                        words.add(word);
                    }
                    Map<String, Object> out = new LinkedHashMap<>();
                    out.put("text", t.text());
                    out.put("words", words);
                    s.result = Json.stringify(out);
                },
                model.driver());
        return s;
    }

    /** jinfer_stream_cancel: from any thread; the producer stops at its next check. */
    void cancel() {
        cancelled = true;
    }

    /**
     * JINFER_OK with text or samples set, JINFER_DONE, or throws the generation's failure. A
     * cancelled stream drains to its end (the producer's result, with what was counted so far) and
     * reports JINFER_ECANCELLED once; after that it is finished. Synchronized so that a second
     * thread stepping the same stream (against the header's contract) waits for DONE instead of
     * blocking forever on an emptied queue.
     */
    synchronized int step() throws InterruptedException {
        if (finished) return Errors.DONE;
        Object next = chunks.take();
        while (cancelled && next != END && !(next instanceof Throwable)) next = chunks.take();
        if (next == END) {
            finished = true;
            if (cancelled) throw new Errors.Failure(Errors.ECANCELLED, "the stream was cancelled");
            return Errors.DONE;
        }
        if (next instanceof Throwable t) {
            finished = true;
            if (t instanceof RuntimeException r) throw r;
            throw new IllegalStateException(t.getMessage(), t);
        }
        if (next instanceof float[] pcm) {
            samples = pcm;
            text = "";
        } else {
            text = (String) next;
            samples = null;
        }
        return Errors.OK;
    }

    // ---- the borrowed C views ----

    CCharPointer cText() {
        if (cText != null) cText.close();
        cText = CTypeConversion.toCString(text);
        return cText.get();
    }

    CCharPointer cResult() {
        if (result == null) throw Errors.invalid("the stream has not finished");
        if (cResult != null) cResult.close();
        cResult = CTypeConversion.toCString(result);
        return cResult.get();
    }

    /** The chunk's samples copied to C memory, NULL when the chunk is text. */
    CFloatPointer cSamples() {
        freeSamples();
        if (samples == null) return WordFactory.nullPointer();
        CFloatPointer out =
                UnmanagedMemory.malloc(WordFactory.unsigned(Math.max(1, samples.length) * 4L));
        for (int i = 0; i < samples.length; i++) out.write(i, samples[i]);
        cSamples = out.rawValue();
        return out;
    }

    private void freeSamples() {
        if (cSamples != 0) UnmanagedMemory.free(WordFactory.pointer(cSamples));
        cSamples = 0;
    }

    void close() {
        boolean early = !finished;
        cancelled = true;
        try {
            while (!finished) step();
        } catch (Exception ignored) {
            // a cancelled pass may end with an aborted result; the stream is gone either way
        }
        if (cText != null) cText.close();
        if (cResult != null) cResult.close();
        freeSamples();
        model.release();
        if (early) Native.collect();
    }
}
