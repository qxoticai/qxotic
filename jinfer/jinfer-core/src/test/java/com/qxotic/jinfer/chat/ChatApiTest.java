package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.util.ArrayList;
import java.util.List;

/** Harness for the chat API core: Batch.prepare policy and the TurnTemplate default encode. */
public final class ChatApiTest {

    static int failures;

    static void check(boolean ok, String what) {
        if (!ok) {
            failures++;
            System.out.println("FAIL: " + what);
        } else {
            System.out.println("ok:   " + what);
        }
    }

    static Batch toks(int... ids) {
        return Batch.prefill(ids);
    }

    static int[] ids(Batch b) {
        return ((Batch.Input.Tokens) b.input()).ids();
    }

    public static void main(String[] args) {
        prepareMergesAdjacentTokenBatches();
        prepareSplitsAtCapacity();
        prepareIsolatesEmbeddings();
        prepareRejectsOversizedBidirectional();
        turnTemplateDefaultEncode();
        messageBasics();
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("ChatApiTest: all passed");
    }

    static void prepareMergesAdjacentTokenBatches() {
        List<Batch> out = Batch.prepare(List.of(toks(1, 2), toks(3), toks(4, 5)), 16);
        check(out.size() == 1, "merge: single fused batch");
        check(
                java.util.Arrays.equals(ids(out.get(0)), new int[] {1, 2, 3, 4, 5}),
                "merge: fused ids in order");
    }

    static void prepareSplitsAtCapacity() {
        List<Batch> out = Batch.prepare(List.of(toks(1, 2, 3), toks(4, 5, 6, 7)), 3);
        check(out.size() == 3, "split: 7 ids at cap 3 -> 3 batches");
        check(
                ids(out.get(0)).length == 3
                        && ids(out.get(1)).length == 3
                        && ids(out.get(2)).length == 1,
                "split: 3+3+1");
        check(ids(out.get(2))[0] == 7, "split: last id survives");
    }

    static void prepareIsolatesEmbeddings() {
        FloatTensor rows = FloatTensor.allocateF32(4 * 8);
        Batch media = Batch.embeddings(rows, 4, true);
        List<Batch> out = Batch.prepare(List.of(toks(1, 2), toks(3), media, toks(4), toks(5)), 16);
        check(out.size() == 3, "isolate: [tokens][media][tokens]");
        check(
                java.util.Arrays.equals(ids(out.get(0)), new int[] {1, 2, 3}),
                "isolate: pre-media run fused");
        check(out.get(1) == media, "isolate: media passes through untouched");
        check(
                java.util.Arrays.equals(ids(out.get(2)), new int[] {4, 5}),
                "isolate: post-media run fused");
    }

    static void prepareRejectsOversizedBidirectional() {
        FloatTensor rows = FloatTensor.allocateF32(8 * 4);
        boolean threw = false;
        try {
            Batch.prepare(List.of(Batch.embeddings(rows, 8, true)), 4);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "oversized bidirectional block throws");
        // causal embeddings of any size pass through (the model may stream them)
        List<Batch> out = Batch.prepare(List.of(Batch.embeddings(rows, 8, false)), 4);
        check(
                out.size() == 1 && out.get(0).count() == 8,
                "oversized causal embeddings pass through");
    }

    /** A toy TurnTemplate: start = [9], turn = [role.length, part count], prompt = [7]. */
    static final class ToyTemplate implements TurnTemplate {
        @Override
        public List<Batch> encodeTurn(Message m) {
            return List.of(Batch.prefill(new int[] {m.role().name().length(), m.content().size()}));
        }

        @Override
        public List<Batch> conversationStart() {
            return List.of(Batch.prefill(new int[] {9}));
        }

        @Override
        public List<Batch> generationPrompt(boolean thinking) {
            return List.of(Batch.prefill(new int[] {thinking ? 8 : 7}));
        }

        @Override
        public List<Batch> closeTurn() {
            return List.of(Batch.prefill(new int[] {5}));
        }
    }

    static void turnTemplateDefaultEncode() {
        TurnTemplate t = new ToyTemplate();
        List<Batch> out = t.encode(List.of(Message.system("s"), Message.user("hi")));
        var flat = new ArrayList<Integer>();
        for (Batch b : out) for (int id : ids(b)) flat.add(id);
        check(flat.equals(List.of(9, 6, 1, 4, 1)), "default encode = start + turns concatenated");
    }

    static void messageBasics() {
        Message m = Message.user("hello");
        check(
                m.role().equals(Role.USER) && m.text().equals("hello"),
                "user factory + text projection");
        boolean threw = false;
        try {
            new Role("");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "empty role rejected");
        check(Message.assistant("x").role().name().equals("assistant"), "assistant factory");
    }
}
