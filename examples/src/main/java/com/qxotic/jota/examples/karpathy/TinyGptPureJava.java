package com.qxotic.jota.examples.karpathy;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class TinyGptPureJava {
    private static final Random RNG = new Random(42);

    private static final int N_EMBD = 16;
    private static final int N_HEAD = 4;
    private static final int N_LAYER = 1;
    private static final int BLOCK_SIZE = 16;
    private static final int HEAD_DIM = N_EMBD / N_HEAD;

    private static final double LEARNING_RATE = 0.01;
    private static final double BETA1 = 0.85;
    private static final double BETA2 = 0.99;
    private static final double EPS_ADAM = 1e-8;
    private static final int NUM_STEPS = 1000;

    private static final double TEMPERATURE = 0.5;
    private static final int NUM_SAMPLES = 20;

    private static final String INPUT_FILE = "input.txt";
    private static final String NAMES_URL =
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";

    private static final LinkedHashMap<String, Value> STATE = new LinkedHashMap<>();
    private static final List<Value> PARAMS = new ArrayList<>();

    private static char[] uchars;
    private static Map<Character, Integer> stoi;
    private static int bos;
    private static int vocabSize;

    private TinyGptPureJava() {}

    public static void main(String[] args) throws Exception {
        List<String> docs = loadDocs();
        System.out.println("num docs: " + docs.size());

        buildTokenizer(docs);
        System.out.println("vocab size: " + vocabSize);

        initParameters();
        long nParams = 0;
        for (Value p : PARAMS) {
            nParams += p.size();
        }
        System.out.println("num params: " + nParams);

        train(docs);
        infer();
    }

    private static List<String> loadDocs() throws IOException, InterruptedException {
        Path inputPath = Path.of(INPUT_FILE);
        if (!Files.exists(inputPath)) {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest req = HttpRequest.newBuilder(URI.create(NAMES_URL)).GET().build();
            client.send(req, HttpResponse.BodyHandlers.ofFile(inputPath));
        }

        List<String> lines = Files.readAllLines(inputPath);
        List<String> docs = new ArrayList<>();
        for (String line : lines) {
            String t = line.trim();
            if (!t.isEmpty()) {
                docs.add(t);
            }
        }
        Collections.shuffle(docs, RNG);
        return docs;
    }

    private static void buildTokenizer(List<String> docs) {
        List<Character> chars = new ArrayList<>();
        boolean[] seen = new boolean[65536];
        for (String doc : docs) {
            for (int i = 0; i < doc.length(); i++) {
                char c = doc.charAt(i);
                if (!seen[c]) {
                    seen[c] = true;
                    chars.add(c);
                }
            }
        }
        chars.sort(Character::compareTo);
        uchars = new char[chars.size()];
        for (int i = 0; i < chars.size(); i++) {
            uchars[i] = chars.get(i);
        }

        stoi = new HashMap<>();
        for (int i = 0; i < uchars.length; i++) {
            stoi.put(uchars[i], i);
        }
        bos = uchars.length;
        vocabSize = uchars.length + 1;
    }

    private static void initParameters() {
        STATE.clear();
        PARAMS.clear();

        putParam("wte", matrix(vocabSize, N_EMBD, 0.08));
        putParam("wpe", matrix(BLOCK_SIZE, N_EMBD, 0.08));
        putParam("lm_head", matrix(vocabSize, N_EMBD, 0.08));

        for (int i = 0; i < N_LAYER; i++) {
            putParam("layer" + i + ".attn_wq", matrix(N_EMBD, N_EMBD, 0.08));
            putParam("layer" + i + ".attn_wk", matrix(N_EMBD, N_EMBD, 0.08));
            putParam("layer" + i + ".attn_wv", matrix(N_EMBD, N_EMBD, 0.08));
            putParam("layer" + i + ".attn_wo", matrix(N_EMBD, N_EMBD, 0.08));
            putParam("layer" + i + ".mlp_fc1", matrix(4 * N_EMBD, N_EMBD, 0.08));
            putParam("layer" + i + ".mlp_fc2", matrix(N_EMBD, 4 * N_EMBD, 0.08));
        }
    }

    private static void putParam(String key, Value tensor) {
        STATE.put(key, tensor);
        PARAMS.add(tensor);
    }

    private static Value matrix(int rows, int cols, double std) {
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = RNG.nextGaussian() * std;
        }
        return Value.parameter(data, rows, cols);
    }

    private static Value linear(Value x, Value w) {
        return w.matVec(x);
    }

    private static Value softmax(Value logits) {
        double maxVal = logits.data[0];
        for (int i = 1; i < logits.size(); i++) {
            maxVal = Math.max(maxVal, logits.data[i]);
        }
        Value exps = logits.sub(maxVal).exp();
        Value total = exps.sum();
        return exps.div(total);
    }

    private static Value rmsnorm(Value x) {
        Value ms = x.mul(x).sum().div(x.size());
        Value scale = ms.add(1e-5).pow(-0.5);
        return x.mul(scale);
    }

    private static Value gpt(int tokenId, int posId, List<Value>[] keys, List<Value>[] values) {
        Value tokEmb = STATE.get("wte").row(tokenId);
        Value posEmb = STATE.get("wpe").row(posId);
        Value x = rmsnorm(tokEmb.add(posEmb));

        for (int li = 0; li < N_LAYER; li++) {
            Value xResidual = x;
            x = rmsnorm(x);

            Value q = linear(x, STATE.get("layer" + li + ".attn_wq"));
            Value k = linear(x, STATE.get("layer" + li + ".attn_wk"));
            Value v = linear(x, STATE.get("layer" + li + ".attn_wv"));
            keys[li].add(k);
            values[li].add(v);

            Value[] headOuts = new Value[N_HEAD];
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                Value qh = q.slice(hs, HEAD_DIM);
                int tLen = keys[li].size();

                Value[] attnLogits = new Value[tLen];
                for (int t = 0; t < tLen; t++) {
                    Value kh = keys[li].get(t).slice(hs, HEAD_DIM);
                    attnLogits[t] = qh.mul(kh).sum().div(Math.sqrt(HEAD_DIM));
                }

                Value attnWeights = softmax(Value.stackScalars(attnLogits));
                Value head = Value.zeros(HEAD_DIM);
                for (int t = 0; t < tLen; t++) {
                    Value vh = values[li].get(t).slice(hs, HEAD_DIM);
                    Value wt = attnWeights.pick(t);
                    head = head.add(vh.mul(wt));
                }
                headOuts[h] = head;
            }

            Value xAttn = Value.concat(headOuts);
            x = linear(xAttn, STATE.get("layer" + li + ".attn_wo"));
            x = x.add(xResidual);

            xResidual = x;
            x = rmsnorm(x);
            x = linear(x, STATE.get("layer" + li + ".mlp_fc1")).relu();
            x = linear(x, STATE.get("layer" + li + ".mlp_fc2"));
            x = x.add(xResidual);
        }

        return linear(x, STATE.get("lm_head"));
    }

    private static void train(List<String> docs) {
        List<double[]> m = new ArrayList<>(PARAMS.size());
        List<double[]> v = new ArrayList<>(PARAMS.size());
        for (Value p : PARAMS) {
            m.add(new double[p.size()]);
            v.add(new double[p.size()]);
        }

        for (int step = 0; step < NUM_STEPS; step++) {
            String doc = docs.get(step % docs.size());
            int[] tokens = tokenizeWithBos(doc);
            int n = Math.min(BLOCK_SIZE, tokens.length - 1);

            @SuppressWarnings("unchecked")
            List<Value>[] keys = new ArrayList[N_LAYER];
            @SuppressWarnings("unchecked")
            List<Value>[] values = new ArrayList[N_LAYER];
            for (int li = 0; li < N_LAYER; li++) {
                keys[li] = new ArrayList<>();
                values[li] = new ArrayList<>();
            }

            Value loss = Value.scalar(0.0);
            for (int posId = 0; posId < n; posId++) {
                int tokenId = tokens[posId];
                int targetId = tokens[posId + 1];
                Value logits = gpt(tokenId, posId, keys, values);
                Value probs = softmax(logits);
                Value lossT = probs.pick(targetId).log().neg();
                loss = loss.add(lossT);
            }
            loss = loss.mul(1.0 / n);

            loss.backward();

            double lrT = LEARNING_RATE * (1.0 - (double) step / NUM_STEPS);
            for (int i = 0; i < PARAMS.size(); i++) {
                Value p = PARAMS.get(i);
                double[] mi = m.get(i);
                double[] vi = v.get(i);
                for (int j = 0; j < p.size(); j++) {
                    mi[j] = BETA1 * mi[j] + (1.0 - BETA1) * p.grad[j];
                    vi[j] = BETA2 * vi[j] + (1.0 - BETA2) * p.grad[j] * p.grad[j];
                    double mHat = mi[j] / (1.0 - Math.pow(BETA1, step + 1));
                    double vHat = vi[j] / (1.0 - Math.pow(BETA2, step + 1));
                    p.data[j] -= lrT * mHat / (Math.sqrt(vHat) + EPS_ADAM);
                    p.grad[j] = 0.0;
                }
            }

            System.out.printf("step %4d / %4d | loss %.4f%n", step + 1, NUM_STEPS, loss.scalar());
        }
    }

    private static int[] tokenizeWithBos(String doc) {
        int[] tokens = new int[doc.length() + 2];
        tokens[0] = bos;
        for (int i = 0; i < doc.length(); i++) {
            Integer id = stoi.get(doc.charAt(i));
            if (id == null) {
                throw new IllegalStateException("Unknown character in input: " + doc.charAt(i));
            }
            tokens[i + 1] = id;
        }
        tokens[tokens.length - 1] = bos;
        return tokens;
    }

    private static void infer() {
        System.out.println("\n--- inference (new, hallucinated names) ---");
        for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++) {
            @SuppressWarnings("unchecked")
            List<Value>[] keys = new ArrayList[N_LAYER];
            @SuppressWarnings("unchecked")
            List<Value>[] values = new ArrayList[N_LAYER];
            for (int li = 0; li < N_LAYER; li++) {
                keys[li] = new ArrayList<>();
                values[li] = new ArrayList<>();
            }

            int tokenId = bos;
            StringBuilder sample = new StringBuilder();
            for (int posId = 0; posId < BLOCK_SIZE; posId++) {
                Value logits = gpt(tokenId, posId, keys, values);
                Value probs = softmax(logits.div(TEMPERATURE));
                tokenId = weightedSample(probs.data, RNG);
                if (tokenId == bos) {
                    break;
                }
                sample.append(uchars[tokenId]);
            }
            System.out.printf("sample %2d: %s%n", sampleIdx + 1, sample);
        }
    }

    private static int weightedSample(double[] weights, Random rnd) {
        double total = 0.0;
        for (double w : weights) {
            total += w;
        }
        double r = rnd.nextDouble() * total;
        double c = 0.0;
        for (int i = 0; i < weights.length; i++) {
            c += weights[i];
            if (r <= c) {
                return i;
            }
        }
        return weights.length - 1;
    }

    @FunctionalInterface
    private interface Backward {
        void apply(Value out);
    }

    private static final class Value {
        final int[] shape;
        final double[] data;
        final double[] grad;
        final List<Value> parents;
        Backward backward;

        private Value(int[] shape, double[] data, List<Value> parents, Backward backward) {
            this.shape = shape;
            this.data = data;
            this.grad = new double[data.length];
            this.parents = parents;
            this.backward = backward;
        }

        static Value parameter(double[] data, int... shape) {
            checkSize(data.length, shape);
            return new Value(shape.clone(), data, List.of(), null);
        }

        static Value scalar(double v) {
            return new Value(new int[] {1}, new double[] {v}, List.of(), null);
        }

        static Value zeros(int n) {
            return new Value(new int[] {n}, new double[n], List.of(), null);
        }

        static Value stackScalars(Value[] scalars) {
            double[] outData = new double[scalars.length];
            for (int i = 0; i < scalars.length; i++) {
                if (scalars[i].size() != 1) {
                    throw new IllegalArgumentException("stackScalars expects scalar tensors");
                }
                outData[i] = scalars[i].data[0];
            }
            Value out = new Value(new int[] {scalars.length}, outData, List.of(scalars), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < scalars.length; i++) {
                            scalars[i].grad[0] += o.grad[i];
                        }
                    });
            return out;
        }

        static Value concat(Value[] parts) {
            int total = 0;
            for (Value p : parts) {
                if (p.shape.length != 1) {
                    throw new IllegalArgumentException("concat expects 1D tensors");
                }
                total += p.size();
            }
            double[] outData = new double[total];
            int cursor = 0;
            for (Value p : parts) {
                System.arraycopy(p.data, 0, outData, cursor, p.size());
                cursor += p.size();
            }
            Value out = new Value(new int[] {total}, outData, List.of(parts), null);
            out.setBackward(
                    o -> {
                        int c = 0;
                        for (Value p : parts) {
                            for (int i = 0; i < p.size(); i++) {
                                p.grad[i] += o.grad[c++];
                            }
                        }
                    });
            return out;
        }

        int size() {
            return data.length;
        }

        double scalar() {
            if (size() != 1) {
                throw new IllegalStateException("Tensor is not scalar");
            }
            return data[0];
        }

        Value add(Value other) {
            return binaryOp(other, (a, b) -> a + b, (a, b) -> 1.0, (a, b) -> 1.0);
        }

        Value add(double c) {
            return add(scalar(c));
        }

        Value sub(Value other) {
            return add(other.neg());
        }

        Value sub(double c) {
            return add(-c);
        }

        Value mul(Value other) {
            return binaryOp(other, (a, b) -> a * b, (a, b) -> b, (a, b) -> a);
        }

        Value mul(double c) {
            return mul(scalar(c));
        }

        Value div(Value other) {
            return mul(other.pow(-1.0));
        }

        Value div(double c) {
            return mul(1.0 / c);
        }

        Value pow(double exponent) {
            double[] outData = new double[size()];
            for (int i = 0; i < size(); i++) {
                outData[i] = Math.pow(data[i], exponent);
            }
            Value out = new Value(shape.clone(), outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < size(); i++) {
                            grad[i] += exponent * Math.pow(data[i], exponent - 1.0) * o.grad[i];
                        }
                    });
            return out;
        }

        Value log() {
            double[] outData = new double[size()];
            for (int i = 0; i < size(); i++) {
                outData[i] = Math.log(data[i]);
            }
            Value out = new Value(shape.clone(), outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < size(); i++) {
                            grad[i] += (1.0 / data[i]) * o.grad[i];
                        }
                    });
            return out;
        }

        Value exp() {
            double[] outData = new double[size()];
            for (int i = 0; i < size(); i++) {
                outData[i] = Math.exp(data[i]);
            }
            Value out = new Value(shape.clone(), outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < size(); i++) {
                            grad[i] += o.data[i] * o.grad[i];
                        }
                    });
            return out;
        }

        Value relu() {
            double[] outData = new double[size()];
            for (int i = 0; i < size(); i++) {
                outData[i] = Math.max(0.0, data[i]);
            }
            Value out = new Value(shape.clone(), outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < size(); i++) {
                            grad[i] += (data[i] > 0 ? 1.0 : 0.0) * o.grad[i];
                        }
                    });
            return out;
        }

        Value neg() {
            return mul(-1.0);
        }

        Value sum() {
            double s = 0.0;
            for (double d : data) {
                s += d;
            }
            Value out = new Value(new int[] {1}, new double[] {s}, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < size(); i++) {
                            grad[i] += o.grad[0];
                        }
                    });
            return out;
        }

        Value pick(int index) {
            if (shape.length != 1) {
                throw new IllegalArgumentException("pick expects 1D tensor");
            }
            double[] outData = new double[] {data[index]};
            Value out = new Value(new int[] {1}, outData, List.of(this), null);
            out.setBackward(o -> grad[index] += o.grad[0]);
            return out;
        }

        Value row(int index) {
            if (shape.length != 2) {
                throw new IllegalArgumentException("row expects 2D tensor");
            }
            int rows = shape[0];
            int cols = shape[1];
            if (index < 0 || index >= rows) {
                throw new IllegalArgumentException("row index out of range");
            }
            double[] outData = new double[cols];
            System.arraycopy(data, index * cols, outData, 0, cols);
            Value out = new Value(new int[] {cols}, outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        int base = index * cols;
                        for (int i = 0; i < cols; i++) {
                            grad[base + i] += o.grad[i];
                        }
                    });
            return out;
        }

        Value slice(int start, int length) {
            if (shape.length != 1) {
                throw new IllegalArgumentException("slice expects 1D tensor");
            }
            if (start < 0 || length < 0 || start + length > size()) {
                throw new IllegalArgumentException("slice out of range");
            }
            double[] outData = new double[length];
            System.arraycopy(data, start, outData, 0, length);
            Value out = new Value(new int[] {length}, outData, List.of(this), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < length; i++) {
                            grad[start + i] += o.grad[i];
                        }
                    });
            return out;
        }

        Value matVec(Value vec) {
            if (shape.length != 2 || vec.shape.length != 1) {
                throw new IllegalArgumentException("matVec expects [out,in] x [in]");
            }
            int outN = shape[0];
            int inN = shape[1];
            if (vec.size() != inN) {
                throw new IllegalArgumentException("matVec shape mismatch");
            }
            double[] outData = new double[outN];
            for (int o = 0; o < outN; o++) {
                double sum = 0.0;
                int base = o * inN;
                for (int i = 0; i < inN; i++) {
                    sum += data[base + i] * vec.data[i];
                }
                outData[o] = sum;
            }
            Value out = new Value(new int[] {outN}, outData, List.of(this, vec), null);
            out.setBackward(
                    o -> {
                        for (int outIdx = 0; outIdx < outN; outIdx++) {
                            int base = outIdx * inN;
                            for (int inIdx = 0; inIdx < inN; inIdx++) {
                                grad[base + inIdx] += o.grad[outIdx] * vec.data[inIdx];
                                vec.grad[inIdx] += o.grad[outIdx] * data[base + inIdx];
                            }
                        }
                    });
            return out;
        }

        void backward() {
            if (size() != 1) {
                throw new IllegalStateException("backward() expects scalar root");
            }
            List<Value> topo = new ArrayList<>();
            IdentityHashMap<Value, Boolean> visited = new IdentityHashMap<>();
            buildTopo(this, visited, topo);
            grad[0] = 1.0;
            for (int i = topo.size() - 1; i >= 0; i--) {
                Value v = topo.get(i);
                if (v.backward != null) {
                    v.backward.apply(v);
                }
            }
        }

        private void setBackward(Backward backwardFn) {
            this.backward = backwardFn;
        }

        private static void buildTopo(
                Value v, IdentityHashMap<Value, Boolean> visited, List<Value> topo) {
            if (visited.containsKey(v)) {
                return;
            }
            visited.put(v, true);
            for (Value parent : v.parents) {
                buildTopo(parent, visited, topo);
            }
            topo.add(v);
        }

        private Value binaryOp(
                Value other, DoubleBinary f, DoubleBinary dSelf, DoubleBinary dOther) {
            boolean sameShape = sameShape(this.shape, other.shape);
            boolean selfScalar = this.size() == 1;
            boolean otherScalar = other.size() == 1;

            if (!sameShape && !selfScalar && !otherScalar) {
                throw new IllegalArgumentException(
                        "Unsupported broadcast: "
                                + shapeToString(shape)
                                + " vs "
                                + shapeToString(other.shape));
            }

            int outSize;
            int[] outShape;
            if (sameShape) {
                outSize = this.size();
                outShape = this.shape.clone();
            } else if (selfScalar) {
                outSize = other.size();
                outShape = other.shape.clone();
            } else {
                outSize = this.size();
                outShape = this.shape.clone();
            }

            double[] outData = new double[outSize];
            for (int i = 0; i < outSize; i++) {
                double a = selfScalar ? this.data[0] : this.data[i];
                double b = otherScalar ? other.data[0] : other.data[i];
                outData[i] = f.apply(a, b);
            }

            Value out = new Value(outShape, outData, List.of(this, other), null);
            out.setBackward(
                    o -> {
                        for (int i = 0; i < outSize; i++) {
                            double a = selfScalar ? this.data[0] : this.data[i];
                            double b = otherScalar ? other.data[0] : other.data[i];
                            double g = o.grad[i];
                            if (selfScalar) {
                                this.grad[0] += dSelf.apply(a, b) * g;
                            } else {
                                this.grad[i] += dSelf.apply(a, b) * g;
                            }
                            if (otherScalar) {
                                other.grad[0] += dOther.apply(a, b) * g;
                            } else {
                                other.grad[i] += dOther.apply(a, b) * g;
                            }
                        }
                    });
            return out;
        }

        private static void checkSize(int dataLen, int[] shape) {
            int expected = 1;
            for (int dim : shape) {
                expected *= dim;
            }
            if (dataLen != expected) {
                throw new IllegalArgumentException(
                        "Data length " + dataLen + " does not match shape size " + expected);
            }
        }

        private static boolean sameShape(int[] a, int[] b) {
            if (a.length != b.length) {
                return false;
            }
            for (int i = 0; i < a.length; i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }

        private static String shapeToString(int[] s) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < s.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(s[i]);
            }
            sb.append(']');
            return sb.toString();
        }
    }

    @FunctionalInterface
    private interface DoubleBinary {
        double apply(double a, double b);
    }
}
