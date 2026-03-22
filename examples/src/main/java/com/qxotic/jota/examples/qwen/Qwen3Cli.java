package com.qxotic.jota.examples.qwen;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.examples.qwen.chat.ChatFormat;
import com.qxotic.jota.examples.qwen.chat.ChatMLFormat;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.GPT2Tokenizer;
import com.qxotic.tokenizers.impl.IntPair;
import com.qxotic.tokenizers.impl.RegexSplitter;
import com.qxotic.tokenizers.impl.VocabularyImpl;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public final class Qwen3Cli {

    private static final String QWEN2_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Environment env =
                Environment.withDefaultDevice(Environment.current().nativeDevice());
        Environment.with(
                env,
                () -> {
                    try {
                        run(options);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    return null;
                });
    }

    private static void run(Options options) throws IOException {
        try (LoadedModel loaded = LoadedModel.load(options.modelPath)) {
            if (options.interactive) {
                runInteractive(options, loaded);
            } else {
                runSinglePrompt(options, loaded);
            }
        }
    }

    private static void runSinglePrompt(Options options, LoadedModel loaded) {
        if (options.prompt == null || options.prompt.isBlank()) {
            throw new IllegalArgumentException("--prompt is required in non-interactive mode");
        }
        ChatFormat format = loaded.chatFormat;
        Qwen3Model model = loaded.model;
        Sampler sampler =
                new Sampler(
                        loaded.config.vocabularySize,
                        options.temperature,
                        options.topP,
                        options.seed);

        List<Integer> tokens = new ArrayList<>();
        format.beginOfText().ifPresent(tokens::add);
        if (options.systemPrompt != null && !options.systemPrompt.isBlank()) {
            append(
                    tokens,
                    format.encodeMessage(
                            new ChatFormat.Message(ChatFormat.SYSTEM, options.systemPrompt)));
        }
        append(
                tokens,
                format.encodeMessage(new ChatFormat.Message(ChatFormat.USER, options.prompt)));
        append(tokens, format.encodeHeader(ChatFormat.ASSISTANT));

        float[] logits = null;
        for (int token : tokens) {
            if (token < 0 || token >= loaded.config.vocabularySize) {
                throw new IllegalArgumentException(
                        "Prompt token out of range: "
                                + token
                                + " (vocab="
                                + loaded.config.vocabularySize
                                + ")");
            }
            logits = model.forward(token);
        }

        IntSequence.Builder generated = IntSequence.newBuilder();
        for (int i = 0; i < options.maxTokens; i++) {
            int next = sampler.sample(logits);
            generated.add(next);
            if (options.stream) {
                System.out.print(format.stream(IntSequence.of(next)));
            }
            logits = model.forward(next);
            if (format.stopTokens().contains(next)) {
                break;
            }
        }
        if (options.stream) {
            System.out.println();
        } else {
            IntSequence out = generated.build();
            if (!out.isEmpty() && format.stopTokens().contains(out.getLast())) {
                out = out.subSequence(0, out.length() - 1);
            }
            System.out.println(format.echo(out));
        }
    }

    private static void runInteractive(Options options, LoadedModel loaded) {
        ChatFormat format = loaded.chatFormat;
        Qwen3Model model = loaded.model;
        Sampler sampler =
                new Sampler(
                        loaded.config.vocabularySize,
                        options.temperature,
                        options.topP,
                        options.seed);

        List<Integer> conversation = new ArrayList<>();
        format.beginOfText().ifPresent(conversation::add);
        if (options.systemPrompt != null && !options.systemPrompt.isBlank()) {
            append(
                    conversation,
                    format.encodeMessage(
                            new ChatFormat.Message(ChatFormat.SYSTEM, options.systemPrompt)));
        }

        int consumed = 0;
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            String user = scanner.nextLine();
            if ("exit".equalsIgnoreCase(user) || "quit".equalsIgnoreCase(user)) {
                break;
            }

            append(
                    conversation,
                    format.encodeMessage(new ChatFormat.Message(ChatFormat.USER, user)));
            append(conversation, format.encodeHeader(ChatFormat.ASSISTANT));

            float[] logits = null;
            for (int i = consumed; i < conversation.size(); i++) {
                int token = conversation.get(i);
                if (token < 0 || token >= loaded.config.vocabularySize) {
                    throw new IllegalArgumentException(
                            "Conversation token out of range: "
                                    + token
                                    + " (vocab="
                                    + loaded.config.vocabularySize
                                    + ")");
                }
                logits = model.forward(conversation.get(i));
            }
            consumed = conversation.size();

            IntSequence.Builder generated = IntSequence.newBuilder();
            for (int i = 0; i < options.maxTokens; i++) {
                int next = sampler.sample(logits);
                generated.add(next);
                conversation.add(next);
                consumed++;
                System.out.print(format.stream(IntSequence.of(next)));
                logits = model.forward(next);
                if (format.stopTokens().contains(next)) {
                    break;
                }
            }
            System.out.println();
        }
    }

    private static void append(List<Integer> out, IntSequence sequence) {
        for (int i = 0; i < sequence.length(); i++) {
            out.add(sequence.intAt(i));
        }
    }

    private static final class LoadedModel implements AutoCloseable {
        final Qwen3Config config;
        public final Tokenizer tokenizer;
        final ChatFormat chatFormat;
        final Qwen3Model model;
        final Arena arena;
        final FileChannel channel;

        private LoadedModel(
                Qwen3Config config,
                Tokenizer tokenizer,
                ChatFormat chatFormat,
                Qwen3Model model,
                Arena arena,
                FileChannel channel) {
            this.config = config;
            this.tokenizer = tokenizer;
            this.chatFormat = chatFormat;
            this.model = model;
            this.arena = arena;
            this.channel = channel;
        }

        static LoadedModel load(Path modelPath) throws IOException {
            GGUF gguf = GGUF.read(modelPath);
            String arch = gguf.getValue(String.class, "general.architecture");
            if (!"qwen3".equals(arch)) {
                throw new IllegalArgumentException("Expected qwen3 architecture, got: " + arch);
            }

            Tokenizer tokenizer = loadTokenizer(gguf);
            Qwen3Config config = loadConfig(gguf, tokenizer.vocabulary().size());
            ChatFormat chatFormat = new ChatMLFormat(tokenizer);

            FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ);
            Arena arena = Arena.ofShared();
            long tensorBase = gguf.getTensorDataOffset();
            Qwen3Weights weights = loadWeights(gguf, config, tensorBase, channel, arena);
            return new LoadedModel(
                    config, tokenizer, chatFormat, new Qwen3Model(config, weights), arena, channel);
        }

        @Override
        public void close() throws IOException {
            arena.close();
            channel.close();
        }
    }

    private static Tokenizer loadTokenizer(GGUF gguf) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes =
                gguf.containsKey("tokenizer.ggml.token_type")
                        ? gguf.getValue(int[].class, "tokenizer.ggml.token_type")
                        : null;
        String[] mergeLines = gguf.getValue(String[].class, "tokenizer.ggml.merges");
        VocabularyImpl vocabulary = new VocabularyImpl(tokens, null, tokenTypes);
        List<IntPair> merges =
                Arrays.stream(mergeLines)
                        .map(line -> line.split(" "))
                        .map(parts -> new IntPair(vocabulary.id(parts[0]), vocabulary.id(parts[1])))
                        .toList();
        Splitter splitter = RegexSplitter.create(QWEN2_PATTERN);
        return new GPT2Tokenizer(vocabulary, Normalizer.IDENTITY, splitter, merges);
    }

    private static Qwen3Config loadConfig(GGUF gguf, int vocabSize) {
        String arch = "qwen3";
        int dim = gguf.getValue(int.class, arch + ".embedding_length");
        int nHeads = gguf.getValue(int.class, arch + ".attention.head_count");
        int kvHeads =
                gguf.containsKey(arch + ".attention.head_count_kv")
                        ? gguf.getValue(int.class, arch + ".attention.head_count_kv")
                        : nHeads;
        int keyHeadSize =
                gguf.containsKey(arch + ".attention.key_length")
                        ? gguf.getValue(int.class, arch + ".attention.key_length")
                        : dim / nHeads;
        int valueHeadSize =
                gguf.containsKey(arch + ".attention.value_length")
                        ? gguf.getValue(int.class, arch + ".attention.value_length")
                        : dim / nHeads;
        if (keyHeadSize != valueHeadSize) {
            throw new UnsupportedOperationException(
                    "key/value head size mismatch is not supported yet");
        }
        return new Qwen3Config(
                dim,
                gguf.getValue(int.class, arch + ".feed_forward_length"),
                gguf.getValue(int.class, arch + ".block_count"),
                nHeads,
                kvHeads,
                keyHeadSize,
                gguf.getValue(int.class, arch + ".context_length"),
                vocabSize,
                gguf.containsKey(arch + ".attention.layer_norm_rms_epsilon")
                        ? gguf.getValue(float.class, arch + ".attention.layer_norm_rms_epsilon")
                        : 1e-5f,
                gguf.containsKey(arch + ".rope.freq_base")
                        ? gguf.getValue(float.class, arch + ".rope.freq_base")
                        : 10000f,
                true);
    }

    private static Qwen3Weights loadWeights(
            GGUF gguf, Qwen3Config cfg, long tensorBase, FileChannel channel, Arena arena) {
        Tensor tokenEmbd = mapTensor(gguf, "token_embd.weight", tensorBase, channel, arena);
        Tensor output =
                gguf.containsTensor("output.weight")
                        ? mapTensor(gguf, "output.weight", tensorBase, channel, arena)
                        : tokenEmbd;

        Tensor[] attnNorm = new Tensor[cfg.nLayers];
        Tensor[] wq = new Tensor[cfg.nLayers];
        Tensor[] wk = new Tensor[cfg.nLayers];
        Tensor[] wv = new Tensor[cfg.nLayers];
        Tensor[] wqNorm = new Tensor[cfg.nLayers];
        Tensor[] wkNorm = new Tensor[cfg.nLayers];
        Tensor[] bq = new Tensor[cfg.nLayers];
        Tensor[] bk = new Tensor[cfg.nLayers];
        Tensor[] bv = new Tensor[cfg.nLayers];
        Tensor[] wo = new Tensor[cfg.nLayers];
        Tensor[] ffnNorm = new Tensor[cfg.nLayers];
        Tensor[] wGate = new Tensor[cfg.nLayers];
        Tensor[] wDown = new Tensor[cfg.nLayers];
        Tensor[] wUp = new Tensor[cfg.nLayers];
        for (int i = 0; i < cfg.nLayers; i++) {
            attnNorm[i] =
                    mapTensor(gguf, "blk." + i + ".attn_norm.weight", tensorBase, channel, arena);
            wq[i] = mapTensor(gguf, "blk." + i + ".attn_q.weight", tensorBase, channel, arena);
            wk[i] = mapTensor(gguf, "blk." + i + ".attn_k.weight", tensorBase, channel, arena);
            wv[i] = mapTensor(gguf, "blk." + i + ".attn_v.weight", tensorBase, channel, arena);
            wo[i] = mapTensor(gguf, "blk." + i + ".attn_output.weight", tensorBase, channel, arena);
            ffnNorm[i] =
                    mapTensor(gguf, "blk." + i + ".ffn_norm.weight", tensorBase, channel, arena);
            wGate[i] = mapTensor(gguf, "blk." + i + ".ffn_gate.weight", tensorBase, channel, arena);
            wDown[i] = mapTensor(gguf, "blk." + i + ".ffn_down.weight", tensorBase, channel, arena);
            wUp[i] = mapTensor(gguf, "blk." + i + ".ffn_up.weight", tensorBase, channel, arena);

            if (gguf.containsTensor("blk." + i + ".attn_q_norm.weight")) {
                wqNorm[i] =
                        mapTensor(
                                gguf,
                                "blk." + i + ".attn_q_norm.weight",
                                tensorBase,
                                channel,
                                arena);
            }
            if (gguf.containsTensor("blk." + i + ".attn_k_norm.weight")) {
                wkNorm[i] =
                        mapTensor(
                                gguf,
                                "blk." + i + ".attn_k_norm.weight",
                                tensorBase,
                                channel,
                                arena);
            }
            if (gguf.containsTensor("blk." + i + ".attn_q.bias")) {
                bq[i] = mapTensor(gguf, "blk." + i + ".attn_q.bias", tensorBase, channel, arena);
            }
            if (gguf.containsTensor("blk." + i + ".attn_k.bias")) {
                bk[i] = mapTensor(gguf, "blk." + i + ".attn_k.bias", tensorBase, channel, arena);
            }
            if (gguf.containsTensor("blk." + i + ".attn_v.bias")) {
                bv[i] = mapTensor(gguf, "blk." + i + ".attn_v.bias", tensorBase, channel, arena);
            }
        }

        Tensor outNorm = mapTensor(gguf, "output_norm.weight", tensorBase, channel, arena);
        float[] ropeScales =
                gguf.containsTensor("rope_freqs.weight")
                        ? toFloatArray(
                                mapTensor(gguf, "rope_freqs.weight", tensorBase, channel, arena))
                        : null;
        RopeTables rope =
                RopeTables.precompute(cfg.contextLength, cfg.headDim, cfg.ropeTheta, ropeScales);

        return new Qwen3Weights(
                tokenEmbd, output, outNorm, attnNorm, wq, wk, wv, wqNorm, wkNorm, bq, bk, bv, wo,
                ffnNorm, wGate, wDown, wUp, rope);
    }

    private static Tensor mapTensor(
            GGUF gguf, String tensorName, long tensorBase, FileChannel channel, Arena arena) {
        TensorEntry e = gguf.getTensor(tensorName);
        if (e == null) {
            throw new IllegalArgumentException("Missing tensor: " + tensorName);
        }
        if (e.ggmlType() != GGMLType.F32) {
            throw new UnsupportedOperationException(
                    "Only F32 tensors are supported for now: "
                            + tensorName
                            + " is "
                            + e.ggmlType());
        }
        long byteSize = e.ggmlType().byteSizeForShape(e.shape());
        try {
            MemorySegment segment =
                    channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            tensorBase + e.offset(),
                            byteSize,
                            arena);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
            Shape shape =
                    e.shape().length == 2
                            ? Shape.of(e.shape()[1], e.shape()[0])
                            : Shape.flat(e.shape());
            MemoryView<MemorySegment> view =
                    MemoryView.of(memory, 0, DataType.FP32, Layout.rowMajor(shape));
            return Tensor.of(view);
        } catch (IOException ex) {
            throw new RuntimeException("Failed to map tensor: " + tensorName, ex);
        }
    }

    private static float[] toFloatArray(Tensor tensor) {
        MemoryView<?> view = tensor.materialize();
        MemoryView<MemorySegment> host = ensureHostView(view);
        int size = Math.toIntExact(host.shape().size());
        float[] out = new float[size];
        @SuppressWarnings("unchecked")
        MemoryAccess<MemorySegment> access =
                (MemoryAccess<MemorySegment>)
                        Environment.nativeMemoryDomain().directAccess();
        for (int i = 0; i < size; i++) {
            long off = Indexing.linearToOffset(host, i);
            out[i] = access.readFloat(host.memory(), off);
        }
        return out;
    }

    private static MemoryView<MemorySegment> ensureHostView(MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> host = (MemoryView<MemorySegment>) view;
            return host;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.runtimeFor(view.memory().device()).memoryDomain();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryDomain<MemorySegment> hostDomain = Environment.nativeMemoryDomain();
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        hostDomain.memoryAllocator().allocateMemory(view.dataType(), view.shape()),
                        view.dataType(),
                        view.layout());
        MemoryDomain.copy(srcDomain, srcView, hostDomain, hostView);
        return hostView;
    }

    private record Qwen3Config(
            int dim,
            int ffnDim,
            int nLayers,
            int nHeads,
            int nKvHeads,
            int headDim,
            int contextLength,
            int vocabularySize,
            float rmsEps,
            float ropeTheta,
            boolean ropeIsNeox) {}

    private record Qwen3Weights(
            Tensor tokenEmbd,
            Tensor output,
            Tensor outputNorm,
            Tensor[] attnNorm,
            Tensor[] wq,
            Tensor[] wk,
            Tensor[] wv,
            Tensor[] wqNorm,
            Tensor[] wkNorm,
            Tensor[] bq,
            Tensor[] bk,
            Tensor[] bv,
            Tensor[] wo,
            Tensor[] ffnNorm,
            Tensor[] wGate,
            Tensor[] wDown,
            Tensor[] wUp,
            RopeTables rope) {}

    private static final class Qwen3Model {
        private final Qwen3Config cfg;
        private final Qwen3Weights w;
        private final float[][] keyCacheArr;
        private final float[][] valueCacheArr;
        private final MemoryView<?>[] keyCache;
        private final MemoryView<?>[] valueCache;
        private final Tensor normReduceOnes;
        private final Tensor normBroadcastOnes;
        private final Tensor headReduceOnes;
        private final Tensor headBroadcastOnes;
        private final Tensor headsOnes;
        private final Tensor contextReduceOnes;
        private final Tensor contextBroadcastOnes;
        private int position;

        Qwen3Model(Qwen3Config cfg, Qwen3Weights w) {
            this.cfg = cfg;
            this.w = w;
            int kvDim = cfg.nKvHeads * cfg.headDim;
            this.keyCacheArr = new float[cfg.nLayers][cfg.contextLength * kvDim];
            this.valueCacheArr = new float[cfg.nLayers][cfg.contextLength * kvDim];
            this.keyCache = new MemoryView<?>[cfg.nLayers];
            this.valueCache = new MemoryView<?>[cfg.nLayers];
            MemoryDomain<?> domain =
                    Environment.runtimeFor(Environment.current().defaultDevice()).memoryDomain();
            for (int i = 0; i < cfg.nLayers; i++) {
                this.keyCache[i] =
                        MemoryView.of(
                                domain.memoryAllocator()
                                        .allocateMemory(
                                                DataType.FP32, Shape.of(cfg.contextLength, kvDim)),
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.contextLength, kvDim)));
                this.valueCache[i] =
                        MemoryView.of(
                                domain.memoryAllocator()
                                        .allocateMemory(
                                                DataType.FP32, Shape.of(cfg.contextLength, kvDim)),
                                DataType.FP32,
                                Layout.rowMajor(Shape.of(cfg.contextLength, kvDim)));
            }
            float[] normOnes = new float[cfg.dim];
            Arrays.fill(normOnes, 1f);
            this.normReduceOnes = Tensor.of(normOnes, Shape.of(cfg.dim, 1));
            this.normBroadcastOnes = Tensor.of(normOnes, Shape.of(1, cfg.dim));
            float[] headOnes = new float[cfg.headDim];
            Arrays.fill(headOnes, 1f);
            this.headReduceOnes = Tensor.of(headOnes, Shape.of(cfg.headDim, 1));
            this.headBroadcastOnes = Tensor.of(headOnes, Shape.of(1, cfg.headDim));
            float[] headsVector = new float[cfg.nHeads];
            Arrays.fill(headsVector, 1f);
            this.headsOnes = Tensor.of(headsVector, Shape.of(cfg.nHeads, 1));
            float[] contextOnes = new float[cfg.contextLength];
            Arrays.fill(contextOnes, 1f);
            this.contextReduceOnes = Tensor.of(contextOnes, Shape.of(cfg.contextLength, 1));
            this.contextBroadcastOnes = Tensor.of(contextOnes, Shape.of(1, cfg.contextLength));
            this.position = 0;
        }

        float[] forward(int token) {
            if (position >= cfg.contextLength) {
                throw new IllegalStateException("Context length exceeded: " + cfg.contextLength);
            }

            Tensor x = w.tokenEmbd.slice(0, token, token + 1);
            int kvDim = cfg.nKvHeads * cfg.headDim;
            int qDim = cfg.nHeads * cfg.headDim;

            for (int layer = 0; layer < cfg.nLayers; layer++) {
                Tensor xb = rmsNorm(x, w.attnNorm[layer]);
                Tensor q = project(xb, w.wq[layer]);
                Tensor k = project(xb, w.wk[layer]);
                Tensor v = project(xb, w.wv[layer]);

                if (w.bq[layer] != null) {
                    q = q.add(w.bq[layer].view(Shape.of(1, qDim)));
                }
                if (w.bk[layer] != null) {
                    k = k.add(w.bk[layer].view(Shape.of(1, kvDim)));
                }
                if (w.bv[layer] != null) {
                    v = v.add(w.bv[layer].view(Shape.of(1, kvDim)));
                }

                // TODO: restore q/k per-head norm once tensor path is validated end-to-end.

                // TODO: re-enable once tensor/materialization path is stable with mapped GGUF
                // weights.
                // q = applyRoPE(q, cfg.nHeads, position, w.rope);
                // k = applyRoPE(k, cfg.nKvHeads, position, w.rope);

                float[] kArr = toFloatArray(k);
                float[] vArr = toFloatArray(v);
                System.arraycopy(kArr, 0, keyCacheArr[layer], position * kvDim, kvDim);
                System.arraycopy(vArr, 0, valueCacheArr[layer], position * kvDim, kvDim);

                Tensor attnOut = attention(layer, q, position + 1);
                Tensor attnProj = project(attnOut, w.wo[layer]);
                x = x.add(attnProj);

                Tensor ffnIn = rmsNorm(x, w.ffnNorm[layer]);
                Tensor gate = project(ffnIn, w.wGate[layer]);
                Tensor up = project(ffnIn, w.wUp[layer]);
                Tensor act = gate.silu().multiply(up);
                Tensor down = project(act, w.wDown[layer]);
                x = x.add(down);
            }

            Tensor out = rmsNorm(x, w.outputNorm);
            Tensor logits = project(out, w.output);
            position++;
            return toFloatArray(logits);
        }

        private static Tensor project(Tensor x, Tensor weightOutIn) {
            return x.matmul(weightOutIn.transpose(0, 1));
        }

        private Tensor attention(int layer, Tensor q, int length) {
            int kvMul = cfg.nHeads / cfg.nKvHeads;
            float[] out = new float[cfg.nHeads * cfg.headDim];
            float[] qArr = toFloatArray(q);
            int kvDim = cfg.nKvHeads * cfg.headDim;
            for (int h = 0; h < cfg.nHeads; h++) {
                int kvHead = h / kvMul;
                int kvOffset = kvHead * cfg.headDim;
                float[] scores = new float[length];
                for (int t = 0; t < length; t++) {
                    float dot = 0f;
                    int qBase = h * cfg.headDim;
                    int kvBase = t * kvDim + kvOffset;
                    for (int i = 0; i < cfg.headDim; i++) {
                        dot += qArr[qBase + i] * keyCacheArr[layer][kvBase + i];
                    }
                    scores[t] = (float) (dot / Math.sqrt(cfg.headDim));
                }
                softmaxInPlace(scores);
                int outBase = h * cfg.headDim;
                for (int t = 0; t < length; t++) {
                    float p = scores[t];
                    int kvBase = t * kvDim + kvOffset;
                    for (int i = 0; i < cfg.headDim; i++) {
                        out[outBase + i] += p * valueCacheArr[layer][kvBase + i];
                    }
                }
            }
            return Tensor.of(out, Shape.of(1, cfg.nHeads * cfg.headDim));
        }

        private static void softmaxInPlace(float[] x) {
            float max = Float.NEGATIVE_INFINITY;
            for (float v : x) {
                if (v > max) {
                    max = v;
                }
            }
            double sum = 0;
            for (int i = 0; i < x.length; i++) {
                double e = Math.exp(x[i] - max);
                x[i] = (float) e;
                sum += e;
            }
            for (int i = 0; i < x.length; i++) {
                x[i] /= (float) sum;
            }
        }

        private Tensor applyHeadRmsNorm(Tensor x, int heads, Tensor weight) {
            Tensor xHeads = x.view(Shape.of(heads, cfg.headDim));
            Tensor meanSquare =
                    xHeads.multiply(xHeads).matmul(headReduceOnes).divide((float) cfg.headDim);
            Tensor invStd =
                    meanSquare.add(cfg.rmsEps).sqrt().reciprocal().matmul(headBroadcastOnes);
            Tensor weightExpanded =
                    headsOnes.slice(0, 0, heads).matmul(weight.view(Shape.of(1, cfg.headDim)));
            return xHeads.multiply(invStd)
                    .multiply(weightExpanded)
                    .view(Shape.of(1, heads * cfg.headDim));
        }

        private Tensor applyRoPE(Tensor x, int heads, int pos, RopeTables rope) {
            if (!cfg.ropeIsNeox) {
                throw new UnsupportedOperationException("Only NeoX RoPE style is implemented");
            }
            int half = cfg.headDim / 2;
            float[] in = toFloatArray(x);
            float[] cos = toFloatArray(rope.cos.slice(0, pos, pos + 1));
            float[] sin = toFloatArray(rope.sin.slice(0, pos, pos + 1));
            float[] out = new float[heads * cfg.headDim];
            for (int h = 0; h < heads; h++) {
                int base = h * cfg.headDim;
                for (int i = 0; i < half; i++) {
                    float a = in[base + i];
                    float b = in[base + half + i];
                    float c = cos[i];
                    float s = sin[i];
                    out[base + i] = a * c - b * s;
                    out[base + half + i] = a * s + b * c;
                }
            }
            return Tensor.of(out, Shape.of(1, heads * cfg.headDim));
        }

        private Tensor rmsNorm(Tensor x, Tensor weight) {
            Tensor meanSquare = x.multiply(x).matmul(normReduceOnes).divide((float) cfg.dim);
            Tensor invStd = meanSquare.add(cfg.rmsEps).sqrt().reciprocal();
            Tensor invExpanded = invStd.matmul(normBroadcastOnes);
            return x.multiply(invExpanded).multiply(weight.view(Shape.of(1, cfg.dim)));
        }

        private Tensor softmaxLast(Tensor x, int length) {
            Tensor exp = x.exp();
            Tensor denom = exp.matmul(contextReduceOnes.slice(0, 0, length));
            Tensor denomExpanded = denom.matmul(contextBroadcastOnes.slice(1, 0, length));
            return exp.divide(denomExpanded);
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        private static void copyTensorToView(Tensor src, MemoryView<?> dst) {
            MemoryView<?> srcView = src.materialize();
            MemoryDomain srcDomain =
                    Environment.runtimeFor(srcView.memory().device()).memoryDomain();
            MemoryDomain dstDomain =
                    Environment.runtimeFor(dst.memory().device()).memoryDomain();
            MemoryDomain.copy(srcDomain, srcView, dstDomain, dst);
        }
    }

    private static final class RopeTables {
        final Tensor cos;
        final Tensor sin;

        private RopeTables(Tensor cos, Tensor sin) {
            this.cos = cos;
            this.sin = sin;
        }

        static RopeTables precompute(
                int contextLength, int headDim, float theta, float[] ropeScales) {
            int half = headDim / 2;
            float[] cosValues = new float[contextLength * half];
            float[] sinValues = new float[contextLength * half];
            int idx = 0;
            for (int pos = 0; pos < contextLength; pos++) {
                for (int i = 0; i < headDim; i += 2) {
                    float freq = (float) (1.0 / Math.pow(theta, i / (double) headDim));
                    if (ropeScales != null) {
                        freq /= ropeScales[i / 2];
                    }
                    float v = pos * freq;
                    cosValues[idx] = (float) Math.cos(v);
                    sinValues[idx] = (float) Math.sin(v);
                    idx++;
                }
            }
            return new RopeTables(
                    Tensor.of(cosValues, Shape.of(contextLength, half)),
                    Tensor.of(sinValues, Shape.of(contextLength, half)));
        }
    }

    private static final class Sampler {
        private final int vocab;
        private final float temperature;
        private final float topP;
        private final Random rng;

        Sampler(int vocab, float temperature, float topP, long seed) {
            this.vocab = vocab;
            this.temperature = temperature;
            this.topP = topP;
            this.rng = new Random(seed);
        }

        int sample(float[] logits) {
            if (temperature <= 0f) {
                return argmax(logits);
            }
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocab; i++) {
                if (logits[i] > max) {
                    max = logits[i];
                }
            }
            float[] probs = new float[vocab];
            double sum = 0;
            for (int i = 0; i < vocab; i++) {
                double p = Math.exp((logits[i] - max) / temperature);
                probs[i] = (float) p;
                sum += p;
            }
            for (int i = 0; i < vocab; i++) {
                probs[i] /= (float) sum;
            }
            if (topP > 0 && topP < 1) {
                return sampleTopP(probs, topP);
            }
            return sampleCategorical(probs);
        }

        private int sampleCategorical(float[] probs) {
            float r = rng.nextFloat();
            float c = 0f;
            for (int i = 0; i < probs.length; i++) {
                c += probs[i];
                if (r <= c) {
                    return i;
                }
            }
            return probs.length - 1;
        }

        private int sampleTopP(float[] probs, float p) {
            Integer[] idx = new Integer[probs.length];
            for (int i = 0; i < probs.length; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> Float.compare(probs[b], probs[a]));

            double cum = 0;
            int cut = 0;
            for (int i = 0; i < idx.length; i++) {
                cum += probs[idx[i]];
                cut = i;
                if (cum >= p) {
                    break;
                }
            }
            float renorm = 0f;
            for (int i = 0; i <= cut; i++) {
                renorm += probs[idx[i]];
            }
            float r = rng.nextFloat();
            float c = 0f;
            for (int i = 0; i <= cut; i++) {
                c += probs[idx[i]] / renorm;
                if (r <= c) {
                    return idx[i];
                }
            }
            return idx[cut];
        }

        private static int argmax(float[] x) {
            int best = 0;
            for (int i = 1; i < x.length; i++) {
                if (x[i] > x[best]) {
                    best = i;
                }
            }
            return best;
        }
    }

    private static final class Options {
        final Path modelPath;
        final String prompt;
        final String systemPrompt;
        final boolean interactive;
        final float temperature;
        final float topP;
        final long seed;
        final int maxTokens;
        final boolean stream;

        private Options(
                Path modelPath,
                String prompt,
                String systemPrompt,
                boolean interactive,
                float temperature,
                float topP,
                long seed,
                int maxTokens,
                boolean stream) {
            this.modelPath = modelPath;
            this.prompt = prompt;
            this.systemPrompt = systemPrompt;
            this.interactive = interactive;
            this.temperature = temperature;
            this.topP = topP;
            this.seed = seed;
            this.maxTokens = maxTokens;
            this.stream = stream;
        }

        static Options parse(String[] args) {
            Path modelPath = null;
            String prompt = null;
            String systemPrompt = null;
            boolean interactive = false;
            float temperature = 0.7f;
            float topP = 0.95f;
            long seed = System.nanoTime();
            int maxTokens = 256;
            boolean stream = true;

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                String value;
                if (arg.contains("=")) {
                    String[] parts = arg.split("=", 2);
                    arg = parts[0];
                    value = parts[1];
                } else {
                    value = (i + 1 < args.length) ? args[i + 1] : null;
                }
                switch (arg) {
                    case "--model", "-m" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        modelPath = Path.of(value);
                    }
                    case "--prompt", "-p" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        prompt = value;
                    }
                    case "--system-prompt", "-sp" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        systemPrompt = value;
                    }
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--temperature", "--temp" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        temperature = Float.parseFloat(value);
                    }
                    case "--top-p" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        topP = Float.parseFloat(value);
                    }
                    case "--seed" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        seed = Long.parseLong(value);
                    }
                    case "--max-tokens", "-n" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        maxTokens = Integer.parseInt(value);
                    }
                    case "--stream" -> {
                        if (!arg.contains("=")) {
                            i++;
                        }
                        stream = Boolean.parseBoolean(value);
                    }
                    case "--help", "-h" -> {
                        printUsage();
                        System.exit(0);
                    }
                    default -> throw new IllegalArgumentException("Unknown option: " + arg);
                }
            }
            if (modelPath == null) {
                throw new IllegalArgumentException("--model is required");
            }
            if (!interactive && (prompt == null || prompt.isBlank())) {
                throw new IllegalArgumentException(
                        "--prompt is required unless --interactive is enabled");
            }
            return new Options(
                    modelPath,
                    prompt,
                    systemPrompt,
                    interactive,
                    temperature,
                    topP,
                    seed,
                    maxTokens,
                    stream);
        }

        private static void printUsage() {
            System.out.println("Usage: Qwen3Cli --model <path> [options]");
            System.out.println("  --prompt, -p <text>");
            System.out.println("  --interactive, -i");
            System.out.println("  --system-prompt, -sp <text>");
            System.out.println("  --max-tokens, -n <int>");
            System.out.println("  --temperature <float>");
            System.out.println("  --top-p <float>");
            System.out.println("  --seed <long>");
            System.out.println("  --stream <true|false>");
        }
    }
}
