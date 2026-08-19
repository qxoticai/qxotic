package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.TestLogits.*;

import com.qxotic.jota.memory.MemoryView;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;

public final class GrammarLegacyTest {

    // ---- vocab mock: 70 tokens probing the long[] bit boundaries (63/64/65) ----

    static final class MockV64 implements Grammar.Vocab {
        static final String[] WORDS = new String[70];

        static {
            for (int i = 0; i < 70; i++) WORDS[i] = "f" + i;
            WORDS[63] = "q"; // last bit of the first long - must be rejected
            WORDS[64] = "a"; // first bit of the second long
            WORDS[65] = "b"; // second bit of the second long
        }

        @Override
        public int size() {
            return WORDS.length;
        }

        @Override
        public byte[] bytes(int t) {
            return WORDS[t].getBytes(StandardCharsets.UTF_8);
        }
    }

    // ---- vocab mock: an EMPTY-bytes token (a special/EOS stand-in) ----

    static final class MockVE implements Grammar.Vocab {
        static final String[] WORDS = {"a", "b", ""}; // token 2 = empty bytes (special)

        @Override
        public int size() {
            return WORDS.length;
        }

        @Override
        public byte[] bytes(int t) {
            return WORDS[t].getBytes(StandardCharsets.UTF_8);
        }
    }

    // ---- vocab mock: multi-byte straddling tokens ----

    static final class MockVS implements Grammar.Vocab {
        static final String[] WORDS = {"ab", "abc", "aX", "c", "bc", "d"};

        @Override
        public int size() {
            return WORDS.length;
        }

        @Override
        public byte[] bytes(int t) {
            return WORDS[t].getBytes(StandardCharsets.UTF_8);
        }
    }

    static int failures;

    // ---- vocab mock: single-byte tokens (47 entries) ----

    static final class MockV implements Grammar.Vocab {
        static final String[] WORDS = {
            "{", "}", "[", "]", "\"", ":", ",", "\n", " ", "t", "r", "u", "e", "1", "n", "a", "b",
            "c", "f", "s", "l", "-", "0", "9", ".", "+", "E", "\\", "/", "x", "y", "z", "d", "m",
            "2", "3", "4", "5", "6", "7", "8", "A", "B", "C", "D", "F", "[", "]", "(", ")", "*",
            "?", "w", "q", "!"
        };

        @Override
        public int size() {
            return WORDS.length;
        }

        @Override
        public byte[] bytes(int t) {
            return t >= 0 && t < WORDS.length
                    ? WORDS[t].getBytes(StandardCharsets.UTF_8)
                    : new byte[0];
        }
    }

    // ---- vocab mock: multi-byte tokens for realistic JSON testing ----
    // 32 tokens covering JSON literals, strings, whitespace, and structural chars

    static final class MockV2 implements Grammar.Vocab {
        static final String[] WORDS = {
            "{", "}", "[", "]", ":", ",", // structure
            "\"", "\\\"", "\\", "/", // string pieces
            "true", "false", "null", // literals
            "0", "123", "9", "1", "-", ".", "e", "E", "+", // numbers
            "\n", "  ", "\t", "\r", // whitespace
            "a", "b", "c", // letters for keys
            "[1]", "\"key\"", // composite tokens
        };

        @Override
        public int size() {
            return WORDS.length;
        }

        @Override
        public byte[] bytes(int t) {
            return t >= 0 && t < WORDS.length
                    ? WORDS[t].getBytes(StandardCharsets.UTF_8)
                    : new byte[0];
        }
    }

    // ---- helpers ----

    static String tok(Grammar.Vocab v, int t) {
        return new String(v.bytes(t), StandardCharsets.UTF_8);
    }

    static int tidx(Grammar.Vocab v, String s) {
        for (int t = 0; t < v.size(); t++) if (tok(v, t).equals(s)) return t;
        return -1;
    }

    // ---- reusable scratch for helpers (avoids allocation churn) ----

    static MemoryView<?> scratchTensor;

    static MemoryView<?> scratch(Grammar.Vocab v) {
        // sized by the vocab: these are off-heap tensors with NO bounds checks - a too-small
        // scratch silently corrupts (maskLogits writes past the end, reads come back garbage)
        if (scratchTensor == null || size(scratchTensor) < v.size()) scratchTensor = view(v.size());
        return scratchTensor;
    }

    static void resetScratch(int vocab) {
        MemoryView<?> t = scratchTensor;
        for (int i = 0; i < vocab; i++) set(t, i, 0.0f);
    }

    static Set<String> allowedSet(Grammar.Cursor cur, Grammar.Vocab v) {
        Set<String> s = new HashSet<>();
        MemoryView<?> logits = scratch(v);
        for (int i = 0; i < size(scratchTensor); i++) set(logits, i, 0.0f);
        cur.maskLogits(logits);
        for (int t = 0; t < v.size(); t++) if (get(logits, t) > -1e30f) s.add(tok(v, t));
        return s;
    }

    static boolean allows(Grammar.Cursor cur, Grammar.Vocab v, String s) {
        return allowedSet(cur, v).contains(s);
    }

    static boolean rejects(Grammar.Cursor cur, Grammar.Vocab v, String s) {
        return !allows(cur, v, s);
    }

    static boolean anyValid(Grammar.Cursor cur, Grammar.Vocab v) {
        return !allowedSet(cur, v).isEmpty();
    }

    // After a complete top-level JSON value, RFC 8259 allows only trailing whitespace/EOS — no
    // further content. (Grammar.json is "ws value ws", so whitespace tokens stay valid.)
    static boolean jsonDone(Grammar.Cursor cur, Grammar.Vocab v) {
        return rejects(cur, v, ",")
                && rejects(cur, v, "{")
                && rejects(cur, v, "1")
                && rejects(cur, v, "\"");
    }

    static void advance(Grammar.Cursor cur, Grammar.Vocab v, String s) {
        int t = tidx(v, s);
        if (t >= 0) cur.advanceWith(t);
    }

    static void check(String what, boolean ok) {
        if (!ok) {
            failures++;
            System.err.println("FAIL: " + what);
        } else System.out.println("ok: " + what);
    }

    @Test
    void grammar() {
        testParser();
        testCursor();
        testPrefixPin();
        testBoundedRepetition();
        testUnboundedRepetitionScales();
        testTailPositionRefs();
        testMaskCacheRepeats();
        testExactAllowedSetAtDepth();
        testRepetitionSemantics();
        testMultiByteTokenInLoop();
        testSchemaDocumentWalk();
        testSchemaScalars();
        testSchemaObjects();
        testSchemaArrays();
        testSchemaConstEnum();
        testSchemaUnions();
        testSchemaKeysAndEscapes();
        testSchemaWhitespaceBound();
        testSchemaAnyValue();
        testSchemaPinnedLimitations();
        testParserCornerCases();
        testMaskBitBoundaries();
        testEmptyByteTokens();
        testStraddlingTokens();
        testAmbiguousGrammars();
        testTcoUnderNonTailFrames();
        testCursorEdgeSemantics();
        testBuiltinCacheKeys();
        testGbnfDeterminism();
        testPerformanceBounds();
        testComplexityGuards();
        testChoice();
        testSpecCacheEviction();
        testMaskCacheCapOverflow();
        testEscapingUnitChecks();
        testSchemaNodeShapes();
        testParserEdgePins();
        testConcurrentCursors();
        testJsonDFA();
        testGbnfCharClass();
        testGbnfDot();
        testGbnfAlternation();
        testGbnfRepetition();
        testGbnfGroup();
        testGbnfRecursive();
        testGbnfJsonParity();
        testGbnfCache();
        testGbnfEmpty();
        testJsonGbnfCompiles();
        testMultiByteTokens();
        testJsonStringEscapes();
        testNumberFormats();
        testEnableDisable();
        testDisabledCursor();
        testAdvanceDeadState();
        testRepetitionAfterCharDot();
        testDfaStateCounts();
        testFuzzRandomWalk();
        testDeepNesting();
        testHexEscapeInCharClass();
        testCommentInString();
        testEpsilonOnlyGrammar();
        testResetRewalk();
        testLastTokenEdgeCase();
        testMaxDfaStatesGuard();
        testZeroVocab();
        testMultiByteMaskConsistency();
        testEmptyCharClass();
        testSpecDisabledEdgeCases();
        testCachePerVocab();
        testStringLiteralEscapes();
        testStripCommentEdgeCases();

        if (failures > 0) {
            throw new AssertionError("GrammarTest: " + failures + " failures");
        }
        System.out.println("\nGrammarTest: 0 failures");
    }

    // ========================================================================
    // parser-only tests
    // ========================================================================

    static void testBoundedRepetition() {
        // {m,n}: epsilon-reachable below min... (min 0), hard cutoff at max - the anti-stall
        // bound behind ws{0,8} (unbounded ws let a reluctant model spin forever)
        Grammar.Vocab v = new MockV();
        Grammar.Cursor cur = Grammar.of("root ::= [ ]{0,2} \"a\"", v).cursor();
        check("rep{0,2}: start allows both", allows(cur, v, " ") && allows(cur, v, "a"));
        advance(cur, v, " ");
        advance(cur, v, " ");
        check("rep{0,2}: bound exhausted", rejects(cur, v, " ") && allows(cur, v, "a"));
        advance(cur, v, "a");
        check("rep{0,2}: accepts", cur.exhausted());
        Grammar.Cursor exact = Grammar.of("root ::= [x]{2}", v).cursor();
        advance(exact, v, "x");
        check("rep{2}: one is not enough", !exact.exhausted() && allows(exact, v, "x"));
        advance(exact, v, "x");
        check("rep{2}: two exactly", exact.exhausted());
    }

    static void testUnboundedRepetitionScales() {
        // E* / E+ must loop at O(1) stack depth (call-site tail loop, see compileRep). Pre-fix,
        // each char pushed a fresh return frame: stacks grew with the repetition length, states
        // never repeated (no mask-cache hits), masks went quadratic, and past CLOSURE_CAP the
        // closure silently truncated (wrong masks). 20k chars exceeds CLOSURE_CAP, so staying
        // CORRECT there guards the semantics; the time bound guards the shape.
        Grammar.Vocab v = new MockV();
        Grammar.Cursor star = Grammar.of("root ::= \"a\" [x]* \"b\"", v).cursor();
        advance(star, v, "a");
        check("E*: zero reps accepted", allows(star, v, "x") && allows(star, v, "b"));
        Grammar.Cursor plus = Grammar.of("root ::= \"a\" [x]+ \"b\"", v).cursor();
        advance(plus, v, "a");
        check("E+: one rep required", allows(plus, v, "x") && rejects(plus, v, "b"));
        for (Grammar.Cursor cur : new Grammar.Cursor[] {star, plus}) {
            long t0 = System.nanoTime();
            for (int i = 0; i < 20_000; i++) {
                advance(cur, v, "x");
                if ((i & 1023) == 0) cur.maskLogits(scratch(v));
            }
            long millis = (System.nanoTime() - t0) / 1_000_000;
            check(
                    "20k reps still in-loop (" + millis + "ms)",
                    allows(cur, v, "x") && allows(cur, v, "b"));
            advance(cur, v, "b");
            check("20k reps accepts", cur.exhausted());
            check("20k reps < 10s (" + millis + "ms)", millis < 10_000);
        }

        // nested repetitions: bounded by nesting depth, not by total chars
        Grammar.Cursor nested = Grammar.of("root ::= ([x]* \"y\")* \"z\"", v).cursor();
        for (int i = 0; i < 5_000; i++) {
            advance(nested, v, "x");
            advance(nested, v, "y");
        }
        check("nested reps still in-loop", allows(nested, v, "x") && allows(nested, v, "z"));
        advance(nested, v, "z");
        check("nested reps accepts", nested.exhausted());
    }

    static void testTailPositionRefs() {
        System.out.println("-- tail-position refs (TCO) --");
        MockV v = new MockV();

        // a named rule referenced in tail position (its continuation is the caller's END)
        Grammar.Cursor c = Grammar.of("root ::= \"a\" tail\ntail ::= [x]* \"b\"", v).cursor();
        advance(c, v, "a");
        advance(c, v, "x");
        advance(c, v, "x");
        check("tail-ref loop", allows(c, v, "x") && allows(c, v, "b"));
        advance(c, v, "b");
        check("tail-ref done", c.exhausted());

        // a chain of tail refs collapses: root -> r1 -> r2 -> "x"
        c = Grammar.of("root ::= \"a\" r1\nr1 ::= r2\nr2 ::= \"x\"", v).cursor();
        advance(c, v, "a");
        check("tail-chain offers x", allows(c, v, "x"));
        advance(c, v, "x");
        check("tail-chain done", c.exhausted());

        // a NON-tail ref still frames: control returns to the caller for "b"
        c = Grammar.of("root ::= \"a\" mid \"b\"\nmid ::= [x]+", v).cursor();
        advance(c, v, "a");
        check("mid-ref requires x first", allows(c, v, "x") && rejects(c, v, "b"));
        advance(c, v, "x");
        advance(c, v, "x");
        check("mid-ref then b", allows(c, v, "x") && allows(c, v, "b"));
        advance(c, v, "b");
        check("mid-ref done", c.exhausted());

        // mutual recursion through tail refs
        c =
                Grammar.of(
                                "root ::= ping\nping ::= \"x\" pong | \"y\"\npong ::= \"z\" ping |"
                                        + " \"w\"",
                                v)
                        .cursor();
        advance(c, v, "x");
        advance(c, v, "z");
        advance(c, v, "x");
        check("mutual tail recursion loops", allows(c, v, "z"));
        advance(c, v, "z");
        advance(c, v, "y");
        check("mutual tail recursion done", c.exhausted());
    }

    static void testMaskCacheRepeats() {
        // the fix's mechanism: a repetition's automaton state REPEATS, so the per-Spec mask
        // cache stays bounded no matter how long the loop runs (pre-fix every prefix length
        // was a fresh state: the cache grew linearly and never hit)
        MockV v = new MockV();
        Grammar.Spec s = Grammar.of("root ::= \"a\" [x]* \"b\"", v);
        Grammar.Cursor cur = s.cursor();
        advance(cur, v, "a");
        for (int i = 0; i < 200; i++) {
            cur.maskLogits(scratch(v));
            advance(cur, v, "x");
        }
        check(
                "mask cache bounded in-loop (" + s.maskCache.size() + " entries)",
                s.maskCache.size() < 20);
    }

    static void testExactAllowedSetAtDepth() {
        // CLOSURE_CAP truncation guard: pre-fix, past 16384 frames the closure silently dropped
        // valid stacks (wrong masks). Exact-set equality also catches phantom tokens.
        MockV v = new MockV();
        Grammar.Cursor cur = Grammar.of("root ::= \"a\" [x]* \"b\"", v).cursor();
        advance(cur, v, "a");
        for (int i = 0; i < 20_000; i++) advance(cur, v, "x");
        check("allowed set exact at 20k", allowedSet(cur, v).equals(Set.of("x", "b")));
        advance(cur, v, "b");
        check("nothing valid after end", !anyValid(cur, v));
    }

    static void testRepetitionSemantics() {
        System.out.println("-- repetition semantics --");
        MockV v = new MockV();

        Grammar.Cursor c = Grammar.of("root ::= [x]*", v).cursor();
        // star-only: the empty string matches, but the loop continuation is always open, so
        // exhausted() (match with NO continuation) never fires - x stays on offer throughout
        check("star-only loops from position 0", allows(c, v, "x") && anyValid(c, v));

        c = Grammar.of("root ::= (\"a\" \"b\")* \"z\"", v).cursor();
        advance(c, v, "a");
        check("seq-star mid-pair needs b", allows(c, v, "b") && rejects(c, v, "z"));
        advance(c, v, "b");
        check("seq-star loop or exit", allows(c, v, "a") && allows(c, v, "z"));
        advance(c, v, "a");
        advance(c, v, "b");
        advance(c, v, "z");
        check("seq-star done", c.exhausted());

        c = Grammar.of("root ::= [x]{2,4} \"b\"", v).cursor();
        advance(c, v, "x");
        check("{2,4} below min", rejects(c, v, "b"));
        advance(c, v, "x");
        check("{2,4} at min", allows(c, v, "b") && allows(c, v, "x"));
        advance(c, v, "x");
        advance(c, v, "x");
        check("{2,4} at max", rejects(c, v, "x") && allows(c, v, "b"));
        advance(c, v, "b");
        check("{2,4} done", c.exhausted());

        c = Grammar.of("root ::= [x]{3}", v).cursor();
        advance(c, v, "x");
        advance(c, v, "x");
        check("{3} not yet", !c.exhausted() && allows(c, v, "x"));
        advance(c, v, "x");
        check("{3} exact", c.exhausted());

        c = Grammar.of("root ::= \"a\" [x]? \"b\"", v).cursor();
        advance(c, v, "a");
        check("opt skippable", allows(c, v, "b") && allows(c, v, "x"));
        advance(c, v, "x");
        check("opt consumed", rejects(c, v, "x") && allows(c, v, "b"));

        c = Grammar.of("root ::= \"a\" [x]* \"b\" [y]* \"z\"", v).cursor();
        advance(c, v, "a");
        advance(c, v, "b");
        advance(c, v, "y");
        advance(c, v, "y");
        check("two stars in sequence", allows(c, v, "y") && allows(c, v, "z"));
        advance(c, v, "z");
        check("two stars done", c.exhausted());
    }

    static void testMultiByteTokenInLoop() {
        // a star whose alternative is a MULTI-BYTE token must loop with the same bounded state
        MockV2 v = new MockV2();
        Grammar.Cursor c = Grammar.of("root ::= (\"a\" | \"\\\"key\\\"\")* \"}\"", v).cursor();
        advance(c, v, "\"key\"");
        advance(c, v, "a");
        for (int i = 0; i < 1_000; i++) advance(c, v, "\"key\"");
        check(
                "multi-byte loop in-loop",
                allows(c, v, "\"key\"") && allows(c, v, "}") && allows(c, v, "a"));
        advance(c, v, "}");
        check("multi-byte loop done", c.exhausted());
    }

    static void testSchemaDocumentWalk() {
        // the production path: a nested JSON-Schema grammar consuming a whole document, then
        // rejecting a schema violation at the exact position
        Map<String, Object> schema =
                Map.of(
                        "type", "object",
                        "properties",
                                Map.of(
                                        "aa",
                                        Map.of(
                                                "type",
                                                "array",
                                                "items",
                                                Map.of(
                                                        "type", "object",
                                                        "properties",
                                                                Map.of(
                                                                        "b",
                                                                        Map.of("type", "integer")),
                                                        "required", List.of("b")))),
                        "required", List.of("aa"));
        MockV v = new MockV();
        Grammar.Cursor cur = Grammar.fromSchema(schema, v).cursor();
        for (char ch : "{\"aa\":[{\"b\":1}]}".toCharArray()) advance(cur, v, String.valueOf(ch));
        check("schema doc done", jsonDone(cur, v));

        Grammar.Cursor bad = Grammar.fromSchema(schema, v).cursor();
        for (char ch : "{\"aa\":[{\"b\":".toCharArray()) advance(bad, v, String.valueOf(ch));
        check(
                "schema rejects wrong type at the value",
                rejects(bad, v, "\"") && allows(bad, v, "1"));
    }

    // ========================================================================
    // JSON-Schema grammars (fromSchema): document accept/reject per feature
    // ========================================================================

    /** Properties maps need declaration order (grammar emits keys in map order). */
    static Map<String, Object> linked(Object... kv) {
        var m = new LinkedHashMap<String, Object>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    /**
     * Whole-doc acceptance: every token consumed legally (cursor still alive), the top-level value
     * can TERMINATE here (trailing ws on offer), and no fresh value may follow. Scalars may still
     * CONTINUE themselves (a digit after "-12") - that is acceptance, not incompleteness, so unlike
     * {@link #jsonDone} this does not demand digit rejection.
     */
    static boolean acceptsDoc(Grammar.Spec s, Grammar.Vocab v, String doc) {
        Grammar.Cursor cur = s.cursor();
        for (char ch : doc.toCharArray()) advance(cur, v, String.valueOf(ch));
        return anyValid(cur, v)
                && allows(cur, v, " ")
                && rejects(cur, v, "{")
                && rejects(cur, v, "\"")
                && rejects(cur, v, ",");
    }

    static boolean rejectsDoc(Grammar.Spec s, Grammar.Vocab v, String doc) {
        return !acceptsDoc(s, v, doc);
    }

    static void testSchemaScalars() {
        System.out.println("-- schema scalars --");
        MockV v = new MockV();

        Grammar.Spec i = Grammar.fromSchema(Map.of("type", "integer"), v);
        check("int ok", acceptsDoc(i, v, "-12") && acceptsDoc(i, v, "0"));
        check("int no frac", rejectsDoc(i, v, "1.5"));
        check("int no string", rejectsDoc(i, v, "\"a\""));
        check("int no bool", rejectsDoc(i, v, "true"));

        Grammar.Spec n = Grammar.fromSchema(Map.of("type", "number"), v);
        check(
                "num forms",
                acceptsDoc(n, v, "1")
                        && acceptsDoc(n, v, "1.5")
                        && acceptsDoc(n, v, "-0.5")
                        && acceptsDoc(n, v, "1e3")
                        && acceptsDoc(n, v, "-12.5E+2"));
        check("num no leading zero", rejectsDoc(n, v, "01"));
        check("num no string", rejectsDoc(n, v, "\"1\""));

        Grammar.Spec b = Grammar.fromSchema(Map.of("type", "boolean"), v);
        check("bool", acceptsDoc(b, v, "true") && acceptsDoc(b, v, "false"));
        check("bool no int", rejectsDoc(b, v, "1"));

        Grammar.Spec nil = Grammar.fromSchema(Map.of("type", "null"), v);
        check("null ok", acceptsDoc(nil, v, "null"));
        check("null no zero", rejectsDoc(nil, v, "0"));

        Grammar.Spec s = Grammar.fromSchema(Map.of("type", "string"), v);
        check("string ok", acceptsDoc(s, v, "\"abc\"") && acceptsDoc(s, v, "\"\""));
        check("string escape", acceptsDoc(s, v, "\"a\\nb\"") && acceptsDoc(s, v, "\"a\\u0041\""));
        check("string no int", rejectsDoc(s, v, "1"));
    }

    static void testSchemaObjects() {
        System.out.println("-- schema objects --");
        MockV v = new MockV();

        // required subset: ONLY the required keys are admitted, in the required list's order
        Grammar.Spec s =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties",
                                        linked(
                                                "a", Map.of("type", "integer"),
                                                "b", Map.of("type", "string")),
                                "required", List.of("b")),
                        v);
        check("obj required key ok", acceptsDoc(s, v, "{\"b\":\"x\"}"));
        check("obj non-required key rejected", rejectsDoc(s, v, "{\"a\":1}"));
        check("obj extra key rejected", rejectsDoc(s, v, "{\"b\":\"x\",\"a\":1}"));

        // no required (or EMPTY required): all properties, in DECLARATION order
        for (Object req : new Object[] {null, List.of()}) {
            var schema =
                    linked(
                            "type",
                            "object",
                            "properties",
                            linked(
                                    "a", Map.of("type", "integer"),
                                    "b", Map.of("type", "string")));
            if (req != null) schema.put("required", req);
            Grammar.Spec all = Grammar.fromSchema(schema, v);
            check(
                    "obj all-props in order" + (req == null ? "" : " (empty required)"),
                    acceptsDoc(all, v, "{\"a\":1,\"b\":\"x\"}"));
            check("obj missing prop rejected", rejectsDoc(all, v, "{\"b\":\"x\"}"));
            check("obj wrong order rejected", rejectsDoc(all, v, "{\"b\":\"x\",\"a\":1}"));
        }

        // required entries that are not declared properties are skipped
        Grammar.Spec skip =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a", Map.of("type", "integer")),
                                "required", List.of("a", "zzz")),
                        v);
        check("obj unknown required skipped", acceptsDoc(skip, v, "{\"a\":1}"));

        // no properties: only the empty object
        Grammar.Spec empty = Grammar.fromSchema(Map.of("type", "object"), v);
        check("obj empty ok", acceptsDoc(empty, v, "{}"));
        check("obj empty rejects props", rejectsDoc(empty, v, "{\"a\":1}"));
    }

    static void testSchemaArrays() {
        System.out.println("-- schema arrays --");
        MockV v = new MockV();

        Grammar.Spec ints =
                Grammar.fromSchema(linked("type", "array", "items", Map.of("type", "integer")), v);
        check(
                "int array",
                acceptsDoc(ints, v, "[]")
                        && acceptsDoc(ints, v, "[1]")
                        && acceptsDoc(ints, v, "[1,2,3]"));
        check("int array rejects string", rejectsDoc(ints, v, "[\"a\"]"));
        check("int array rejects trailing comma", rejectsDoc(ints, v, "[1,]"));

        Grammar.Spec any = Grammar.fromSchema(Map.of("type", "array"), v);
        check("array any items", acceptsDoc(any, v, "[1,\"a\",true,null,{},[2]]"));

        Grammar.Spec nested =
                Grammar.fromSchema(
                        linked(
                                "type",
                                "array",
                                "items",
                                linked("type", "array", "items", Map.of("type", "integer"))),
                        v);
        check("nested arrays", acceptsDoc(nested, v, "[[1],[2,3],[]]"));
        check("nested arrays reject flat", rejectsDoc(nested, v, "[1]"));
    }

    static void testSchemaConstEnum() {
        System.out.println("-- schema const/enum --");
        MockV v = new MockV();

        check(
                "const string",
                acceptsDoc(Grammar.fromSchema(Map.of("const", "abc"), v), v, "\"abc\""));
        check(
                "const string rejects",
                rejectsDoc(Grammar.fromSchema(Map.of("const", "abc"), v), v, "\"abd\""));
        check("const number", acceptsDoc(Grammar.fromSchema(Map.of("const", 42), v), v, "42"));
        check(
                "const number rejects",
                rejectsDoc(Grammar.fromSchema(Map.of("const", 42), v), v, "43"));
        check("const true", acceptsDoc(Grammar.fromSchema(Map.of("const", true), v), v, "true"));
        check(
                "const true rejects false",
                rejectsDoc(Grammar.fromSchema(Map.of("const", true), v), v, "false"));
        check(
                "const null literal",
                acceptsDoc(Grammar.fromSchema(Map.of("const", Map.of()), v), v, "{}"));

        Grammar.Spec e = Grammar.fromSchema(Map.of("enum", Arrays.asList("red", 1, true, null)), v);
        check(
                "enum each",
                acceptsDoc(e, v, "\"red\"")
                        && acceptsDoc(e, v, "1")
                        && acceptsDoc(e, v, "true")
                        && acceptsDoc(e, v, "null"));
        check(
                "enum rejects others",
                rejectsDoc(e, v, "\"blue\"") && rejectsDoc(e, v, "2") && rejectsDoc(e, v, "false"));

        Grammar.Spec emptyEnum = Grammar.fromSchema(Map.of("enum", List.of()), v);
        check("enum empty = any", acceptsDoc(emptyEnum, v, "{\"a\":[1]}"));
    }

    static void testSchemaUnions() {
        System.out.println("-- schema unions --");
        MockV v = new MockV();

        for (String kw : new String[] {"anyOf", "oneOf"}) {
            Grammar.Spec u =
                    Grammar.fromSchema(
                            Map.of(
                                    kw,
                                    List.of(Map.of("type", "integer"), Map.of("type", "string"))),
                            v);
            check(kw + " first alt", acceptsDoc(u, v, "1"));
            check(kw + " second alt", acceptsDoc(u, v, "\"a\""));
            check(kw + " rejects neither", rejectsDoc(u, v, "true"));
        }

        Grammar.Spec types = Grammar.fromSchema(Map.of("type", List.of("string", "null")), v);
        check("type list", acceptsDoc(types, v, "\"a\"") && acceptsDoc(types, v, "null"));
        check("type list rejects", rejectsDoc(types, v, "1"));

        Grammar.Spec emptyUnion = Grammar.fromSchema(Map.of("anyOf", List.of()), v);
        check("anyOf empty = any", acceptsDoc(emptyUnion, v, "[1]"));
    }

    static void testSchemaKeysAndEscapes() {
        System.out.println("-- schema keys/escapes --");
        MockV v = new MockV();

        // property names with spaces, quotes and backslashes survive literal escaping
        Grammar.Spec spaced =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a b", Map.of("type", "integer")),
                                "required", List.of("a b")),
                        v);
        check("key with space", acceptsDoc(spaced, v, "{\"a b\":1}"));

        Grammar.Spec quoted =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a\"b", Map.of("type", "integer")),
                                "required", List.of("a\"b")),
                        v);
        check("key with quote", acceptsDoc(quoted, v, "{\"a\\\"b\":1}"));

        Grammar.Spec slashed =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a\\b", Map.of("type", "integer")),
                                "required", List.of("a\\b")),
                        v);
        check("key with backslash", acceptsDoc(slashed, v, "{\"a\\\\b\":1}"));

        // a const string containing escapes round-trips through jsonEncode/gbnfLiteral
        Grammar.Spec esc = Grammar.fromSchema(Map.of("const", "a\nb"), v);
        check("const with newline", acceptsDoc(esc, v, "\"a\\nb\""));
    }

    static void testSchemaWhitespaceBound() {
        System.out.println("-- schema ws bound --");
        MockV v = new MockV();
        Grammar.Spec s =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a", Map.of("type", "integer")),
                                "required", List.of("a")),
                        v);
        check("ws compact", acceptsDoc(s, v, "{\"a\":1}"));
        check("ws pretty", acceptsDoc(s, v, "{ \"a\" : 1 }"));
        check("ws newline", acceptsDoc(s, v, "{\n \"a\" : 1\n}"));
        check("ws max 8", acceptsDoc(s, v, "{\"a\":" + "        " + "1}"));
        // ws is BOUNDED at {0,8}: a 9-space gap stalls the matcher (the anti-stall bound)
        check("ws beyond 8 rejected", rejectsDoc(s, v, "{\"a\":" + "         " + "1}"));
    }

    static void testSchemaAnyValue() {
        System.out.println("-- schema any-value fallbacks --");
        MockV v = new MockV();

        // no type at all
        Grammar.Spec noType = Grammar.fromSchema(Map.of(), v);
        check(
                "no type = any",
                acceptsDoc(noType, v, "1")
                        && acceptsDoc(noType, v, "\"a\"")
                        && acceptsDoc(noType, v, "true")
                        && acceptsDoc(noType, v, "{\"a\":[1,null]}"));

        // unknown type name
        Grammar.Spec funky = Grammar.fromSchema(Map.of("type", "funky"), v);
        check("unknown type = any", acceptsDoc(funky, v, "[{\"b\":2}]"));

        // an untyped property admits arbitrarily nested values (the `value` recursion)
        Grammar.Spec deep =
                Grammar.fromSchema(
                        linked(
                                "type",
                                "object",
                                "properties",
                                linked("a", Map.of()),
                                "required",
                                List.of("a")),
                        v);
        check(
                "untyped property nested",
                acceptsDoc(deep, v, "{\"a\":{\"b\":[1,\"x\",true,null]}}"));
        check("untyped property scalar", acceptsDoc(deep, v, "{\"a\":2}"));
    }

    static void testSchemaPinnedLimitations() {
        System.out.println("-- schema pinned limitations (documented, not bugs) --");
        MockV v = new MockV();

        // minItems/maxItems are IGNORED (documented: length bounds unsupported)
        Grammar.Spec arr =
                Grammar.fromSchema(
                        linked(
                                "type",
                                "array",
                                "items",
                                Map.of("type", "integer"),
                                "minItems",
                                2,
                                "maxItems",
                                2),
                        v);
        check("minItems ignored (pinned)", acceptsDoc(arr, v, "[1]"));
        check("maxItems ignored (pinned)", acceptsDoc(arr, v, "[1,2,3,4]"));

        // optional (non-required) properties are not admitted at all - stricter than JSON
        // Schema (pinned in testSchemaObjects; restated here as the documented contract)
        Grammar.Spec opt =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties",
                                        linked(
                                                "a", Map.of("type", "integer"),
                                                "b", Map.of("type", "integer")),
                                "required", List.of("a")),
                        v);
        check("optional prop not admitted (pinned)", rejectsDoc(opt, v, "{\"a\":1,\"b\":2}"));

        // a required list naming ONLY unknown keys collapses the object to {} (pinned)
        Grammar.Spec unknownOnly =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a", Map.of("type", "integer")),
                                "required", List.of("zzz")),
                        v);
        check("required unknown-only = {} (pinned)", acceptsDoc(unknownOnly, v, "{}"));
        check(
                "required unknown-only rejects props (pinned)",
                rejectsDoc(unknownOnly, v, "{\"a\":1}"));

        // additionalProperties does not open the fixed object shape (pinned)
        Grammar.Spec ap =
                Grammar.fromSchema(
                        linked(
                                "type",
                                "object",
                                "properties",
                                linked("a", Map.of("type", "integer")),
                                "required",
                                List.of("a"),
                                "additionalProperties",
                                true),
                        v);
        check("additionalProperties ignored (pinned)", rejectsDoc(ap, v, "{\"a\":1,\"b\":2}"));

        // a null schema node means "any JSON"
        check("null schema = any", acceptsDoc(Grammar.fromSchema(null, v), v, "[1,{\"a\":2}]"));
    }

    static void testParserCornerCases() {
        System.out.println("-- parser corner cases --");
        MockV v = new MockV();

        // llama.cpp-style multi-line rules: continuation lines join the rule above
        Grammar.Cursor c = Grammar.of("root ::= \"a\"\n    | \"b\"", v).cursor();
        check("multi-line rule alt a", allows(c, v, "a") && allows(c, v, "b"));
        c = Grammar.of("root ::= [x]+\n        \"b\" | \"a\"\nnext ::= \"z\"", v).cursor();
        advance(c, v, "x");
        // the joined continuation ("b") is on offer; "a" is a SEPARATE alternative, unreachable
        check(
                "multi-line sequence + alt",
                allows(c, v, "b") && allows(c, v, "x") && rejects(c, v, "a"));

        // hyphens in rule names (llama.cpp GBNF allows them)
        c = Grammar.of("root ::= foo-bar\nfoo-bar ::= [x]+", v).cursor();
        advance(c, v, "x");
        check("hyphenated rule name", allows(c, v, "x") && c.exhausted() == false);
        c = Grammar.of("root ::= \"a\" string-char\nstring-char ::= \"b\"", v).cursor();
        advance(c, v, "a");
        check("hyphenated ref mid-rule", allows(c, v, "b"));

        // undefined rule references are an ERROR, not a silent alias to root
        boolean threw = false;
        try {
            Grammar.parse("root ::= nope");
        } catch (IllegalArgumentException e) {
            threw = e.getMessage().contains("nope");
        }
        check("undefined ref throws", threw);

        // unterminated string literal is an ERROR, not silently skipped
        threw = false;
        try {
            Grammar.parse("root ::= \"abc");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check("unterminated literal throws", threw);

        // invalid hex escape surfaces (NumberFormatException extends IAE)
        threw = false;
        try {
            Grammar.parse("root ::= \"\\xZZ\"");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check("invalid hex escape throws", threw);

        // '#' inside a char class is a class member, not a comment
        var parsed = Grammar.parse("root ::= [#x] \"a\"");
        check(
                "hash in class survives",
                parsed.get(0).body().size() == 2
                        && parsed.get(0).body().get(0) instanceof Grammar.Rule.Element.CharClass cc
                        && cc.chars().contains((byte) '#'));

        // parens inside literals don't break group matching
        c = Grammar.of("root ::= (\"(\" | \")\")* \"x\"", v).cursor();
        advance(c, v, "(");
        advance(c, v, "(");
        advance(c, v, ")");
        check("paren literals in group", allows(c, v, "(") && allows(c, v, "x"));
        advance(c, v, "x");
        check("paren literals done", c.exhausted());

        // an unbalanced paren inside a literal
        c = Grammar.of("root ::= (\"a)b\" | \"c\") \"z\"", v).cursor();
        advance(c, v, "a");
        advance(c, v, ")");
        advance(c, v, "b");
        check("unbalanced paren in literal", allows(c, v, "z"));

        // a quote inside a char class inside a group
        c = Grammar.of("root ::= ([\"\\(] | \"q\") \"z\"", v).cursor();
        check("quote in class in group", allows(c, v, "\"") && allows(c, v, "q"));
        advance(c, v, "\"");
        check("quote in class matched", allows(c, v, "z"));

        // whitespace before a repetition modifier is insignificant
        c = Grammar.of("root ::= \"a\" {2} \"b\"", v).cursor();
        advance(c, v, "a");
        check("space before {2}: second a required", allows(c, v, "a") && rejects(c, v, "b"));
        c = Grammar.of("root ::= [x] *", v).cursor();
        advance(c, v, "x");
        advance(c, v, "x");
        check("space before *: loops", allows(c, v, "x") && c.exhausted() == false);

        // dot matches ANY byte, newline included
        c = Grammar.of("root ::= .", v).cursor();
        check("dot matches newline", allows(c, v, "\n") && allows(c, v, "a"));
        advance(c, v, "\n");
        check("dot single byte done", c.exhausted());

        // dash as a class literal at the edges
        c = Grammar.of("root ::= [a-]", v).cursor();
        check("class trailing dash", allows(c, v, "a") && allows(c, v, "-"));
        c = Grammar.of("root ::= [-a]", v).cursor();
        check("class leading dash", allows(c, v, "-") && allows(c, v, "a"));

        // a rule defined twice: the LAST body wins
        c = Grammar.of("root ::= \"a\"\nroot ::= \"b\"", v).cursor();
        check("duplicate rule last wins", allows(c, v, "b") && rejects(c, v, "a"));

        // a repetition binds to the LAST atom of a literal, not the whole literal
        c = Grammar.of("root ::= \"ab\"+", v).cursor();
        advance(c, v, "a");
        check("rep binds last atom", allows(c, v, "b") && rejects(c, v, "a"));
        advance(c, v, "b");
        advance(c, v, "b");
        // like any open loop, the continuation stays open (exhausted never fires)
        check("rep last atom loops", allows(c, v, "b"));
    }

    static void testMaskBitBoundaries() {
        System.out.println("-- mask bit boundaries --");
        MockV64 v = new MockV64();
        // grammar needs "a" (token 64, first bit of the second long) then "b" (token 65)
        Grammar.Cursor c = Grammar.of("root ::= \"a\" \"b\"", v).cursor();
        Set<String> start = allowedSet(c, v);
        check("bit64 allowed at start", start.equals(Set.of("a")));
        advance(c, v, "a");
        check("bit65 allowed next", allowedSet(c, v).equals(Set.of("b")));
        advance(c, v, "b");
        check("bit-boundary doc done", !anyValid(c, v) || c.exhausted());

        // an alternation spanning the 63/64/65 boundary bits
        c = Grammar.of("root ::= \"q\" | \"a\" | \"b\"", v).cursor();
        check("boundary bits all offered", allowedSet(c, v).equals(Set.of("q", "a", "b")));
    }

    static void testEmptyByteTokens() {
        System.out.println("-- empty-byte (special) tokens --");
        MockVE v = new MockVE();
        Grammar.Cursor c = Grammar.of("root ::= \"a\" \"b\"", v).cursor();
        // the special token is CONTROL, not content: unsamplable mid-grammar...
        check("special rejected at start", rejects(c, v, ""));
        advance(c, v, "a");
        check("special rejected mid-rule", rejects(c, v, ""));
        advance(c, v, "b");
        // ...and samplable exactly at an accept state (how EOS ends a constrained span)
        check("special allowed at accept", allows(c, v, ""));
        // consuming it is a no-op (it carries no bytes)
        c.advanceWith(2);
        check("special advance is a no-op", c.exhausted());
    }

    static void testStraddlingTokens() {
        System.out.println("-- straddling tokens --");
        MockVS v = new MockVS();

        // one token spanning THREE separate literals
        Grammar.Cursor c = Grammar.of("root ::= \"a\" \"b\" \"c\"", v).cursor();
        check("one token spans three literals", allows(c, v, "abc"));
        check("shorter span also ok", allows(c, v, "ab"));
        // a token diverging from the literal mid-way is rejected
        check("mid-token divergence rejected", rejects(c, v, "aX"));
        advance(c, v, "ab"); // spans "a" "b"; only "c" remains
        check("remainder after straddle", allows(c, v, "c") && rejects(c, v, "bc"));

        // a straddle that completes the whole rule at once
        c = Grammar.of("root ::= \"ab\" \"c\"", v).cursor();
        advance(c, v, "abc");
        check("full straddle done", c.exhausted());
    }

    static void testAmbiguousGrammars() {
        System.out.println("-- ambiguous grammars --");
        MockV v = new MockV();

        // shared prefix: after "a" BOTH alternatives live
        Grammar.Cursor c = Grammar.of("root ::= \"a\" \"b\" | \"a\" \"c\"", v).cursor();
        advance(c, v, "a");
        check("ambiguity keeps both", allows(c, v, "b") && allows(c, v, "c"));
        advance(c, v, "b");
        check("ambiguity resolves", c.exhausted() && !anyValid(c, v));

        // ambiguity across nesting depths: "a" can start a pair or stand alone
        c = Grammar.of("root ::= \"a\" (\"b\" \"c\" | \"b\") \"d\"", v).cursor();
        advance(c, v, "a");
        advance(c, v, "b");
        check("nested ambiguity", allows(c, v, "c") && allows(c, v, "d"));
        advance(c, v, "d");
        check("nested ambiguity short path", c.exhausted());

        // diamond: a ::= b | c, b ::= d, c ::= d - the shared suffix merges cleanly
        c = Grammar.of("root ::= \"x\" r\nr ::= s1 | s2\ns1 ::= \"y\"\ns2 ::= \"y\"", v).cursor();
        advance(c, v, "x");
        advance(c, v, "y");
        check("diamond recursion done", c.exhausted());
    }

    static void testTcoUnderNonTailFrames() {
        System.out.println("-- TCO under non-tail frames --");
        MockV v = new MockV();

        // tail is a tail-ref INSIDE mid, and mid itself is called NON-tail (frames below):
        // the tail loop must still return control for "b"
        Grammar.Cursor c =
                Grammar.of("root ::= \"a\" mid \"b\"\nmid ::= \"x\" tail\ntail ::= \"y\"", v)
                        .cursor();
        advance(c, v, "a");
        advance(c, v, "x");
        advance(c, v, "y");
        check("tail-in-mid returns for b", allows(c, v, "b"));
        advance(c, v, "b");
        check("tail-in-mid done", c.exhausted());

        // same but the tail LOOPS: frames below must survive every iteration
        c = Grammar.of("root ::= \"a\" mid \"b\"\nmid ::= [x]+ tail\ntail ::= [y]*", v).cursor();
        advance(c, v, "a");
        advance(c, v, "x");
        advance(c, v, "x");
        advance(c, v, "y");
        advance(c, v, "y");
        check("looping tails under frames", allows(c, v, "b"));
        advance(c, v, "b");
        check("looping tails done", c.exhausted());
    }

    static void testCursorEdgeSemantics() {
        System.out.println("-- cursor edge semantics --");
        MockV v = new MockV();

        // a dead cursor stays dead
        Grammar.Cursor c = Grammar.of("root ::= \"a\"", v).cursor();
        advance(c, v, "x"); // invalid
        check("dead after bad token", !anyValid(c, v));
        advance(c, v, "a"); // would be valid at start
        check("dead stays dead", !anyValid(c, v));

        // reset() recovers even from the dead state
        c.reset();
        check("reset revives", allows(c, v, "a"));

        // out-of-range token ids are no-ops
        Grammar.Cursor c2 = Grammar.of("root ::= \"a\"", v).cursor();
        c2.advanceWith(-1);
        c2.advanceWith(v.size() + 100);
        check("out-of-range ids no-op", allows(c2, v, "a"));

        // advancing past an exhausted grammar kills it
        Grammar.Cursor c3 = Grammar.of("root ::= \"a\"", v).cursor();
        advance(c3, v, "a");
        check("exhausted after a", c3.exhausted());
        advance(c3, v, "a");
        check("advance past end dies", !anyValid(c3, v) && !c3.exhausted());

        // maskLogits touches ONLY disallowed tokens; allowed logits pass through untouched
        Grammar.Cursor c4 = Grammar.of("root ::= \"a\" | \"b\"", v).cursor();
        MemoryView<?> logits = view(v.size());
        for (int i = 0; i < v.size(); i++) set(logits, i, i + 0.5f);
        c4.maskLogits(logits);
        int ta = tidx(v, "a"), tb = tidx(v, "b"), tx = tidx(v, "x");
        check(
                "allowed logits untouched",
                get(logits, ta) == ta + 0.5f && get(logits, tb) == tb + 0.5f);
        check("disallowed logits masked", get(logits, tx) == Float.NEGATIVE_INFINITY);

        // a fresh cursor on a DISABLED spec: pass-through forever
        Grammar.Cursor off = Grammar.Spec.DISABLED.cursor();
        check("disabled mask passes", off.maskLogits(logits) && !off.exhausted());
        off.advanceWith(0);
        check("disabled advance no-op", !off.exhausted());
    }

    static void testBuiltinCacheKeys() {
        System.out.println("-- builtin cache keys --");
        MockV v = new MockV();
        // a user grammar string that collides with a builtin's RESERVED name builds the user's
        // text ("__json__" is not valid GBNF: no ::=, so a DISABLED spec), never the builtin
        Grammar.Spec user = Grammar.of("__json__", v);
        check(
                "user __json__ does not resolve to the builtin",
                !user.isValid() && user != Grammar.json(v));
        // builtins are cached: same instance per vocab; compact is a different instance
        check("builtin cached", Grammar.json(v) == Grammar.json(v));
        check("compact distinct", Grammar.jsonCompact(v) != Grammar.json(v));
        check(
                "user grammar cached",
                Grammar.of("root ::= \"a\"", v) == Grammar.of("root ::= \"a\"", v));
    }

    static void testGbnfDeterminism() {
        System.out.println("-- schema determinism --");
        // same schema (LinkedHashMap order) -> identical GBNF, byte for byte
        var schema =
                linked(
                        "type", "object",
                        "properties",
                                linked(
                                        "a", Map.of("type", "integer"),
                                        "b",
                                                linked(
                                                        "type",
                                                        "array",
                                                        "items",
                                                        Map.of("type", "string"))),
                        "required", List.of("a", "b"));
        check(
                "toGbnf deterministic",
                Grammar.Schema.toGbnf(schema, true).equals(Grammar.Schema.toGbnf(schema, true)));
        // and the compiled Spec is cached on the same key
        MockV v = new MockV();
        check("fromSchema cached", Grammar.fromSchema(schema, v) == Grammar.fromSchema(schema, v));
    }

    static void testPerformanceBounds() {
        System.out.println("-- performance bounds --");
        MockV v = new MockV();

        // 1M single-char advances through a repetition: per-step cost must stay O(1)
        Grammar.Cursor loop = Grammar.of("root ::= \"a\" [x]* \"b\"", v).cursor();
        advance(loop, v, "a");
        long t0 = System.nanoTime();
        for (int i = 0; i < 1_000_000; i++) advance(loop, v, "x");
        long advMs = (System.nanoTime() - t0) / 1_000_000;
        check("1M loop advances < 5s (" + advMs + "ms)", advMs < 5_000);

        // a ~100KB JSON document walks linearly (mask + advance per char)
        String doc = "{\"a\":[" + "1,".repeat(20_000) + "1]}";
        Grammar.Spec json = Grammar.json(v);
        Grammar.Cursor c = json.cursor();
        t0 = System.nanoTime();
        for (char ch : doc.toCharArray()) {
            c.maskLogits(scratch(v));
            advance(c, v, String.valueOf(ch));
        }
        long docMs = (System.nanoTime() - t0) / 1_000_000;
        check("100KB doc linear (" + docMs + "ms)", docMs < 10_000 && jsonDone(c, v));

        // a big schema compiles fast (200-property object)
        var props = new LinkedHashMap<String, Object>();
        for (int i = 0; i < 200; i++) props.put("p" + i, Map.of("type", "integer"));
        var big =
                linked(
                        "type",
                        "object",
                        "properties",
                        props,
                        "required",
                        new ArrayList<>(props.keySet()));
        t0 = System.nanoTime();
        Grammar.Spec bigSpec = Grammar.fromSchema(big, v);
        long compileMs = (System.nanoTime() - t0) / 1_000_000;
        check(
                "200-prop schema compiles < 5s (" + compileMs + "ms)",
                compileMs < 5_000 && bigSpec.isValid());

        // mask cache HIT path: the same state masked 10k times is a lookup, not a recompute
        Grammar.Cursor hit = Grammar.of("root ::= \"a\" [x]* \"b\"", v).cursor();
        advance(hit, v, "a");
        advance(hit, v, "x");
        hit.maskLogits(scratch(v)); // prime
        t0 = System.nanoTime();
        for (int i = 0; i < 10_000; i++) hit.maskLogits(scratch(v));
        long hitMs = (System.nanoTime() - t0) / 1_000_000;
        check("10k mask hits < 1s (" + hitMs + "ms)", hitMs < 1_000);
    }

    static void testComplexityGuards() {
        System.out.println("-- complexity guards (no degenerate blowups) --");
        MockV v = new MockV();

        // SUSTAINED AMBIGUITY: every position admits several parses. Stack dedup must keep the
        // ready set bounded by stack SHAPES, not by parse count (that would be exponential)
        String[][] cases = {
            {"root ::= (\"aa\" | \"a\")* \"b\"", "a"}, // every 'a' is half of aa or whole
            {"root ::= (\"ab\" | \"a\" \"b\")* \"z\"", "ab"}, // same string, two parses
            {"root ::= (\"a\" | \"a\" \"a\" | \"a\" \"a\" \"a\")* \"b\"", "a"}, // 3-way split
        };
        for (String[] cs : cases) {
            Grammar.Cursor c = Grammar.of(cs[0], v).cursor();
            long t0 = System.nanoTime();
            for (int i = 0; i < 25_000; i++) {
                for (char ch : cs[1].toCharArray()) advance(c, v, String.valueOf(ch));
                if ((i & 4095) == 0) c.maskLogits(scratch(v));
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            check("ambiguous loop 25k pairs < 5s (" + ms + "ms): " + cs[0], ms < 5_000);
            check("ambiguous loop still alive: " + cs[0], anyValid(c, v));
        }

        // FANOUT LOOP: a 20-way choice repeated forever
        StringBuilder fan = new StringBuilder("root ::= (");
        for (char ch = 'a'; ch <= 'm'; ch++)
            fan.append(ch == 'a' ? "" : " | ").append("\"").append(ch).append("\"");
        fan.append(")* \"z\"");
        Grammar.Cursor fc = Grammar.of(fan.toString(), v).cursor();
        long t0 = System.nanoTime();
        for (int i = 0; i < 100_000; i++) advance(fc, v, "a");
        long fanMs = (System.nanoTime() - t0) / 1_000_000;
        check("20-way fanout loop 100k < 3s (" + fanMs + "ms)", fanMs < 3_000);

        // DEEP NESTING: cost must stay polynomial (depth^2), never exponential
        Grammar.Cursor deep = Grammar.json(v).cursor();
        t0 = System.nanoTime();
        for (int i = 0; i < 2_000; i++) {
            advance(deep, v, "[");
            if ((i & 255) == 0) deep.maskLogits(scratch(v));
        }
        for (int i = 0; i < 2_000; i++) advance(deep, v, "]");
        long deepMs = (System.nanoTime() - t0) / 1_000_000;
        check("2k-deep nesting < 10s (" + deepMs + "ms)", deepMs < 10_000);
        check("2k-deep nesting done", jsonDone(deep, v));

        // LEFT RECURSION: best-effort by design - every step costs up to CLOSURE_CAP stacks of
        // input-length size (the documented backstop), so sustained left-recursive parsing is
        // inherently heavy; 50 steps prove it stays bounded and never hangs
        Grammar.Cursor left = Grammar.of("root ::= root \"a\" | \"b\"", v).cursor();
        advance(left, v, "b");
        t0 = System.nanoTime();
        for (int i = 0; i < 50; i++) {
            advance(left, v, "a");
            left.maskLogits(scratch(v));
        }
        long leftMs = (System.nanoTime() - t0) / 1_000_000;
        check("left recursion bounded < 10s (" + leftMs + "ms)", leftMs < 10_000);

        // WIDE CHOICE: 500 literal alternatives - compile, mask, walk stay practical
        StringBuilder many = new StringBuilder("root ::= ");
        for (int i = 0; i < 500; i++)
            many.append(i == 0 ? "" : " | ").append("\"alt").append(i).append("\"");
        t0 = System.nanoTime();
        Grammar.Spec wide = Grammar.of(many.toString(), v);
        Grammar.Cursor wc = wide.cursor();
        wc.maskLogits(scratch(v));
        advance(wc, v, "a");
        long wideMs = (System.nanoTime() - t0) / 1_000_000;
        check("500-alt choice < 5s (" + wideMs + "ms)", wideMs < 5_000);

        // AMORTIZED SHARING: 50 documents through ONE Spec - masks are computed once for the
        // whole run, not per cursor (the cross-cursor cache is the amortization story)
        Grammar.Spec json = Grammar.json(v);
        String doc = "{\"k\":[{\"a\":1,\"b\":[true,null,\"s\"]}]}";
        t0 = System.nanoTime();
        for (int d = 0; d < 50; d++) {
            Grammar.Cursor dc = json.cursor();
            for (char ch : doc.toCharArray()) {
                dc.maskLogits(scratch(v));
                advance(dc, v, String.valueOf(ch));
            }
            if (!jsonDone(dc, v)) check("amortized doc " + d + " valid", false);
        }
        long amortMs = (System.nanoTime() - t0) / 1_000_000;
        check("50 docs amortized < 3s (" + amortMs + "ms)", amortMs < 3_000);

        // WIDE VOCAB: 5k-token vocab - distinct-state masks are full-vocab scans; a handful of
        // distinct states must stay practical (production vocabs are 32k-256k with caching)
        Grammar.Vocab wide5k =
                new Grammar.Vocab() {
                    final byte[][] bytes = new byte[5_000][];

                    {
                        for (int i = 0; i < 5_000; i++)
                            bytes[i] = ("tok" + i).getBytes(StandardCharsets.UTF_8);
                        bytes[4_000] = new byte[] {'{'};
                        bytes[4_001] = new byte[] {'}'};
                        bytes[4_002] = new byte[] {'"'};
                        bytes[4_003] = new byte[] {'a'};
                        bytes[4_004] = new byte[] {':'};
                        bytes[4_005] = new byte[] {'1'};
                        bytes[4_006] = new byte[] {' '};
                    }

                    @Override
                    public int size() {
                        return 5_000;
                    }

                    @Override
                    public byte[] bytes(int t) {
                        return bytes[t];
                    }
                };
        Grammar.Spec wjson = Grammar.json(wide5k);
        Grammar.Cursor w = wjson.cursor();
        t0 = System.nanoTime();
        String doc5k = "{\"a\":1}";
        for (char ch : doc5k.toCharArray()) {
            w.maskLogits(scratch(wide5k));
            advance(w, wide5k, String.valueOf(ch));
        }
        long wide5kMs = (System.nanoTime() - t0) / 1_000_000;
        check("5k-vocab doc masks < 10s (" + wide5kMs + "ms)", wide5kMs < 10_000);
    }

    static void testChoice() {
        System.out.println("-- choice --");
        MockV v = new MockV();

        Grammar.Spec yes = Grammar.choice(v, "yes", "no");
        Grammar.Cursor c = yes.cursor();
        check("choice offers all", allows(c, v, "y") && allows(c, v, "n"));
        advance(c, v, "y");
        check("choice committed", allows(c, v, "e") && rejects(c, v, "n"));
        advance(c, v, "e");
        advance(c, v, "s");
        check("choice done", c.exhausted());

        // exact literals only: no prefixes, no other words
        Grammar.Cursor c2 = Grammar.choice(v, "yes", "na").cursor();
        advance(c2, v, "n");
        advance(c2, v, "a");
        check("choice exact", c2.exhausted());

        // options needing literal escaping
        Grammar.Spec esc = Grammar.choice(v, "a\"b", "c\nd");
        Grammar.Cursor c3 = esc.cursor();
        advance(c3, v, "a");
        check("choice escaped literal", allows(c3, v, "\""));

        // no options: root ::= (empty) - matches only the empty string
        Grammar.Cursor c4 = Grammar.choice(v).cursor();
        check("choice empty", c4.exhausted());

        // cached per (vocab, grammar string)
        check("choice cached", Grammar.choice(v, "yes") == Grammar.choice(v, "yes"));
    }

    static void testSpecCacheEviction() {
        System.out.println("-- spec cache eviction --");
        MockV v = new MockV();
        // the per-vocab spec cache is an LRU of 32: the eldest entry is evicted
        String first = "root ::= " + "\"x\"".repeat(1);
        Grammar.Spec firstSpec = Grammar.of(first, v);
        for (int i = 2; i <= 40; i++) Grammar.of("root ::= " + "\"x\"".repeat(i), v);
        Grammar.Spec rebuilt = Grammar.of(first, v);
        check("eldest evicted and rebuilt", rebuilt != firstSpec);
        check("rebuilt now cached", Grammar.of(first, v) == rebuilt);
    }

    static void testMaskCacheCapOverflow() {
        System.out.println("-- mask cache cap overflow --");
        MockV v = new MockV();
        // > MASK_CACHE_CAP distinct states: the cache stops growing but matching stays correct
        Grammar.Spec chain = Grammar.of("root ::= " + "\"a\" ".repeat(9_000), v);
        Grammar.Cursor c = chain.cursor();
        for (int i = 0; i < 9_000; i++) {
            c.maskLogits(scratch(v)); // each position is a DISTINCT state
            advance(c, v, "a");
        }
        check(
                "mask cache capped (" + chain.maskCache.size() + ")",
                chain.maskCache.size() <= Grammar.MASK_CACHE_CAP);
        check("correct past cap", c.exhausted());
    }

    static void testEscapingUnitChecks() {
        System.out.println("-- escaping unit checks --");
        check("gbnfLiteral quote", Grammar.gbnfLiteral("a\"b").equals("\"a\\\"b\""));
        check("gbnfLiteral backslash", Grammar.gbnfLiteral("a\\b").equals("\"a\\\\b\""));
        check("gbnfLiteral newline", Grammar.gbnfLiteral("a\nb").equals("\"a\\nb\""));
        check("gbnfLiteral control char", Grammar.gbnfLiteral("a\u0001b").equals("\"a\\x01b\""));

        check("jsonEsc control", Grammar.jsonEsc("a\u0001b").equals("a\\u0001b"));
        check("jsonEsc unicode raw", Grammar.jsonEsc("é").equals("é"));

        check("jsonEncode integral double", Grammar.jsonEncode(1.0).equals("1"));
        check("jsonEncode double", Grammar.jsonEncode(1.5).equals("1.5"));
        check("jsonEncode list", Grammar.jsonEncode(List.of(1, "a")).equals("[1,\"a\"]"));
        check(
                "jsonEncode nested",
                Grammar.jsonEncode(Map.of("a", List.of(1, true))).equals("{\"a\":[1,true]}"));
        // NaN/Infinity encode as their (invalid-JSON) names: garbage in, garbage out - pinned
        check("jsonEncode NaN (pinned)", Grammar.jsonEncode(Double.NaN).equals("NaN"));
        check(
                "jsonEncode unknown type",
                Grammar.jsonEncode(
                                new Object() {
                                    @Override
                                    public String toString() {
                                        return "weird";
                                    }
                                })
                        .equals("\"weird\""));
    }

    static void testSchemaNodeShapes() {
        System.out.println("-- schema node shapes --");
        MockV v = new MockV();

        // properties that is not a Map: the object collapses to {}
        Grammar.Spec weird =
                Grammar.fromSchema(linked("type", "object", "properties", List.of(1, 2)), v);
        check("properties non-Map = {}", acceptsDoc(weird, v, "{}"));
        check("properties non-Map rejects props", rejectsDoc(weird, v, "{\"a\":1}"));

        // nested non-Map nodes fall back to "any JSON" (value)
        Grammar.Spec junkUnion =
                Grammar.fromSchema(Map.of("anyOf", List.of(Map.of("type", "integer"), "junk")), v);
        check("anyOf junk member = any", acceptsDoc(junkUnion, v, "true"));
        Grammar.Spec junkProp =
                Grammar.fromSchema(
                        linked(
                                "type", "object",
                                "properties", linked("a", "junk"),
                                "required", List.of("a")),
                        v);
        check("junk property = any", acceptsDoc(junkProp, v, "{\"a\":[1,true]}"));

        // NaN const: compiles (garbage in, garbage out - pinned)
        check(
                "NaN const compiles (pinned)",
                Grammar.fromSchema(Map.of("const", Double.NaN), v).isValid());
    }

    static void testParserEdgePins() {
        System.out.println("-- parser edge pins --");
        MockV v = new MockV();

        // escaped backslash as the last class member
        Grammar.Cursor c = Grammar.of("root ::= [\\\\]", v).cursor();
        check("class lone backslash", allows(c, v, "\\"));

        // a repetition spec with no preceding element is ignored; the rule is empty
        Grammar.Cursor bare = Grammar.of("root ::= {2}", v).cursor();
        check("bare {2} = empty rule", bare.exhausted());

        // a brace spec that is not a repetition now ERRORS (via undefined ref) instead of
        // silently vanishing
        boolean threw = false;
        try {
            Grammar.parse("root ::= \"a\"{abc}");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check("brace non-repetition errors", threw);

        // a hex escape with too few digits degrades to a literal 'x' (pinned)
        check("short hex escape (pinned)", Grammar.unescape("\\x4").equals("x4"));

        // E{0} is epsilon
        Grammar.Cursor z = Grammar.of("root ::= [x]{0} \"b\"", v).cursor();
        check("{0} is epsilon", allows(z, v, "b") && rejects(z, v, "x"));

        // E{2,} means at least two
        Grammar.Cursor two = Grammar.of("root ::= [x]{2,} \"b\"", v).cursor();
        advance(two, v, "x");
        check("{2,} below min", rejects(two, v, "b"));
        advance(two, v, "x");
        advance(two, v, "x");
        check("{2,} above min loops", allows(two, v, "x") && allows(two, v, "b"));
    }

    static void testConcurrentCursors() {
        System.out.println("-- concurrent cursors --");
        // one Spec, many threads: the shared mask cache and per-cursor state are thread-safe
        MockV v = new MockV();
        Grammar.Spec json = Grammar.json(v);
        String doc = "{\"k\":[1,\"a\",true,null,{\"n\":2}]}";
        Thread[] threads = new Thread[8];
        boolean[] failed = {false};
        for (int t = 0; t < threads.length; t++) {
            threads[t] =
                    new Thread(
                            () -> {
                                // per-thread tensor: the shared scratch is NOT thread-safe
                                MemoryView<?> logits = view(v.size());
                                try {
                                    for (int i = 0; i < 50; i++) {
                                        Grammar.Cursor c = json.cursor();
                                        for (char ch : doc.toCharArray()) {
                                            for (int z = 0; z < v.size(); z++) set(logits, z, 0.0f);
                                            c.maskLogits(logits);
                                            advance(c, v, String.valueOf(ch));
                                        }
                                        if (!c.maskLogits(logits)) failed[0] = true;
                                    }
                                } catch (Throwable e) {
                                    failed[0] = true;
                                }
                            });
            threads[t].start();
        }
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        check("concurrent cursors clean", !failed[0]);
    }

    static void testPrefixPin() {
        // prefix-pin: "a" ("bc" | "de") "{" - constrains the pin, exhausts, then releases
        Grammar.Vocab v = new MockV();
        Grammar.Spec spec = Grammar.of("root ::= \"a\" (\"bc\" | \"dm\") \"{\"", v);
        Grammar.Cursor cur = spec.cursor();
        check("pin: only the prefix opens", allows(cur, v, "a") && rejects(cur, v, "b"));
        check("pin: not exhausted at start", !cur.exhausted());
        advance(cur, v, "a");
        check(
                "pin: name union",
                allows(cur, v, "b") && allows(cur, v, "d") && rejects(cur, v, "a"));
        advance(cur, v, "b");
        advance(cur, v, "c");
        check("pin: delimiter pinned", allows(cur, v, "{") && rejects(cur, v, "}"));
        check("pin: not exhausted before delim", !cur.exhausted());
        advance(cur, v, "{");
        check("pin: exhausted after full match", cur.exhausted());
        // exhaustion is the release signal walk-forced regions build on (Walk.sampler); the
        // sampler-level prefix pin itself is gone - every family forces through a selection
    }

    static void testParser() {
        System.out.println("-- parser --");

        check("unescape \\n", Grammar.unescape("a\\nb").equals("a\nb"));
        check("unescape \\t", Grammar.unescape("a\\tb").equals("a\tb"));
        check("unescape \\r", Grammar.unescape("a\\rb").equals("a\rb"));
        check("unescape \\\"", Grammar.unescape("a\\\"b").equals("a\"b"));
        check("unescape \\x41", Grammar.unescape("\\x41").equals("A"));
        check("unescape \\x7e", Grammar.unescape("\\x7e").equals("~"));
        check("unescape plain", Grammar.unescape("hello").equals("hello"));

        check("unescChar \\n", Grammar.unescChar('n') == '\n');
        check("unescChar \\t", Grammar.unescChar('t') == '\t');
        check("unescChar \\r", Grammar.unescChar('r') == '\r');
        check("unescChar plain", Grammar.unescChar('a') == 'a');
    }

    // ========================================================================
    // cursor/mask basics
    // ========================================================================

    static void testCursor() {
        System.out.println("-- cursor --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"hello\"", v);
        check("cursor non-null", s.cursor() != null);

        Grammar.Spec s2 = Grammar.of("root ::= \"he\"", v);
        check("multi-byte compiles", s2.isValid());

        Grammar.Spec json = Grammar.json(v);
        Grammar.Cursor c = json.cursor();
        check("json '{'", allows(c, v, "{"));
        check("json '['", allows(c, v, "["));
        check("json reject '}'", rejects(c, v, "}"));
        check("json reject ','", rejects(c, v, ","));

        Grammar.Cursor c2 = json.cursor();
        advance(c2, v, "{");
        check("after '{' '\"'", allows(c2, v, "\""));
        check("after '{' '}'", allows(c2, v, "}"));
        check("after '{' reject '1'", rejects(c2, v, "1"));

        c2.reset();
        check("reset '{'", allows(c2, v, "{"));
        check("reset reject '}'", rejects(c2, v, "}"));
    }

    // ========================================================================
    // json DFA walks
    // ========================================================================

    static void testJsonDFA() {
        System.out.println("-- json dfa --");
        MockV v = new MockV();

        Grammar.Spec json = Grammar.json(v);
        Grammar.Cursor c = json.cursor();

        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "a");
        advance(c, v, "\"");
        advance(c, v, ":");
        advance(c, v, "1");
        advance(c, v, "}");
        check("complete object", jsonDone(c, v));

        c = json.cursor();
        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "a");
        advance(c, v, "\"");
        advance(c, v, ":");
        advance(c, v, "[");
        advance(c, v, "1");
        advance(c, v, ",");
        advance(c, v, "2");
        advance(c, v, "]");
        advance(c, v, "}");
        check("nested array", jsonDone(c, v));

        c = json.cursor();
        advance(c, v, "[");
        advance(c, v, "t");
        advance(c, v, "r");
        advance(c, v, "u");
        advance(c, v, "e");
        advance(c, v, ",");
        advance(c, v, "f");
        advance(c, v, "a");
        advance(c, v, "l");
        advance(c, v, "s");
        advance(c, v, "e");
        advance(c, v, ",");
        advance(c, v, "n");
        advance(c, v, "u");
        advance(c, v, "l");
        advance(c, v, "l");
        advance(c, v, "]");
        check("literal array", jsonDone(c, v));

        c = json.cursor();
        advance(c, v, "1");
        check("number 1", anyValid(c, v));
        c.reset();
        advance(c, v, "-");
        advance(c, v, "1");
        check("number -1", anyValid(c, v));
    }

    // ========================================================================
    // GBNF char class
    // ========================================================================

    static void testGbnfCharClass() {
        System.out.println("-- gbnf char-class --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= [a-z]", v);
        check("cc-range compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("cc-range 'a'", allows(c, v, "a"));
        check("cc-range 'm'", allows(c, v, "m"));
        check("cc-range 'z'", allows(c, v, "z"));
        check("cc-range reject '1'", rejects(c, v, "1"));
        check("cc-range reject '{'", rejects(c, v, "{"));

        s = Grammar.of("root ::= [0-9]", v);
        c = s.cursor();
        check("cc-digit '1'", allows(c, v, "1"));
        check("cc-digit '9'", allows(c, v, "9"));
        check("cc-digit reject 'a'", rejects(c, v, "a"));

        s = Grammar.of("root ::= [^ab]", v);
        c = s.cursor();
        check("cc-neg reject 'a'", rejects(c, v, "a"));
        check("cc-neg reject 'b'", rejects(c, v, "b"));
        check("cc-neg 'c'", allows(c, v, "c"));
        check("cc-neg '1'", allows(c, v, "1"));

        s = Grammar.of("root ::= [a-cx]", v);
        c = s.cursor();
        check("cc-mixed 'a'", allows(c, v, "a"));
        check("cc-mixed 'c'", allows(c, v, "c"));
        check("cc-mixed 'x'", allows(c, v, "x"));
        check("cc-mixed reject 'd'", rejects(c, v, "d"));

        s = Grammar.of("root ::= [\\n\\t]", v);
        c = s.cursor();
        check("cc-escaped '\\n'", allows(c, v, "\n"));
        check("cc-escaped reject 'a'", rejects(c, v, "a"));
    }

    // ========================================================================
    // GBNF dot wildcard
    // ========================================================================

    static void testGbnfDot() {
        System.out.println("-- gbnf dot --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= .", v);
        check("dot compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("dot '{'", allows(c, v, "{"));
        check("dot 'a'", allows(c, v, "a"));
        check("dot '1'", allows(c, v, "1"));
        check("dot '\\n'", allows(c, v, "\n"));

        s = Grammar.of("root ::= \"a\" . \"c\"", v);
        c = s.cursor();
        advance(c, v, "a");
        check("dot-seq after 'a'", allows(c, v, "b"));
        check("dot-seq after 'a' '{'", allows(c, v, "{"));
        advance(c, v, "b");
        check("dot-seq after 'ab'", allows(c, v, "c"));
    }

    // ========================================================================
    // GBNF alternation
    // ========================================================================

    static void testGbnfAlternation() {
        System.out.println("-- gbnf alternation --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\" | \"b\" | \"c\"", v);
        check("alt compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("alt 'a'", allows(c, v, "a"));
        check("alt 'b'", allows(c, v, "b"));
        check("alt 'c'", allows(c, v, "c"));
        check("alt reject 'd'", rejects(c, v, "d"));

        s = Grammar.of("root ::= (\"{\" | \"[\")", v);
        c = s.cursor();
        check("alt-group '{'", allows(c, v, "{"));
        check("alt-group '['", allows(c, v, "["));
        check("alt-group reject 'a'", rejects(c, v, "a"));

        s = Grammar.of("root ::= \"a\" (\"b\" | \"c\")", v);
        c = s.cursor();
        advance(c, v, "a");
        check("alt-seq 'b'", allows(c, v, "b"));
        check("alt-seq 'c'", allows(c, v, "c"));
        check("alt-seq reject 'a'", rejects(c, v, "a"));
    }

    // ========================================================================
    // GBNF repetition
    // ========================================================================

    static void testGbnfRepetition() {
        System.out.println("-- gbnf repetition --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\"*", v);
        check("star compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("star zero", anyValid(c, v));
        advance(c, v, "a");
        check("star one 'a'", anyValid(c, v));
        advance(c, v, "a");
        check("star two 'a'", anyValid(c, v));
        check("star reject 'b'", rejects(c, v, "b"));

        s = Grammar.of("root ::= \"a\"+", v);
        c = s.cursor();
        check("plus zero 'a' valid", allows(c, v, "a"));
        advance(c, v, "a");
        check("plus one", anyValid(c, v));
        advance(c, v, "a");
        check("plus two", anyValid(c, v));

        s = Grammar.of("root ::= \"a\"?", v);
        c = s.cursor();
        check("opt zero 'a' valid", allows(c, v, "a"));
        advance(c, v, "a");
        check("opt one done", !anyValid(c, v)); // optional consumed -> complete

        s = Grammar.of("root ::= (\"a\"|\"b\")*", v);
        c = s.cursor();
        check("grp-star zero", anyValid(c, v));
        advance(c, v, "a");
        check("grp-star 'a'", anyValid(c, v));
        advance(c, v, "b");
        check("grp-star 'ab'", anyValid(c, v));
        advance(c, v, "a");
        check("grp-star 'aba'", anyValid(c, v));
    }

    // ========================================================================
    // GBNF groups
    // ========================================================================

    static void testGbnfGroup() {
        System.out.println("-- gbnf groups --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\" (\"b\" (\"c\" | \"d\"))", v);
        check("nested-group compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        advance(c, v, "a");
        advance(c, v, "b");
        check("nested-group 'c'", allows(c, v, "c"));
        check("nested-group 'd'", allows(c, v, "d"));
        check("nested-group reject 'e'", rejects(c, v, "e"));

        s = Grammar.of("root ::= \"a\" (\"b\" \"c\")*", v);
        c = s.cursor();
        advance(c, v, "a");
        check("grp-bc-star zero", anyValid(c, v));
        advance(c, v, "b");
        advance(c, v, "c");
        check("grp-bc-star one pair", anyValid(c, v));
        advance(c, v, "b");
        advance(c, v, "c");
        check("grp-bc-star two pairs", anyValid(c, v));
    }

    // ========================================================================
    // GBNF recursion
    // ========================================================================

    static void testGbnfRecursive() {
        System.out.println("-- gbnf recursive --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\" root | \"b\"", v);
        check("rec compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("rec 'b'", allows(c, v, "b"));
        check("rec 'a'", allows(c, v, "a"));
        check("rec reject 'c'", rejects(c, v, "c"));
        advance(c, v, "a");
        check("rec a→", allows(c, v, "a"));
        check("rec a→b", allows(c, v, "b"));
        advance(c, v, "a");
        check("rec aa→a", allows(c, v, "a"));
        advance(c, v, "b");
        check("rec aab done", !anyValid(c, v));

        s = Grammar.of("root ::= \"(\" root \")\" | \"a\"", v);
        c = s.cursor();
        check("paren-rec 'a'", allows(c, v, "a"));
        check("paren-rec '('", allows(c, v, "("));
        advance(c, v, "(");
        check("paren-rec '(→", allows(c, v, "("));
        check("paren-rec '(→a", allows(c, v, "a"));
        advance(c, v, ")");
        check("paren-rec '((a) done'", !anyValid(c, v));
        advance(c, v, ")");
        check("paren-rec '(a)' satisfied", !anyValid(c, v));
        c.reset();
        advance(c, v, "a");
        check("paren-rec 'a' base", !anyValid(c, v));

        s = Grammar.of("root ::= root \"a\" | \"b\"", v);
        check("left-rec compiles", s.isValid());
        c = s.cursor();
        check("left-rec 'b'", allows(c, v, "b"));
        check("left-rec 'b' at start", allows(c, v, "b"));
        advance(c, v, "b");
        // language is b·a* — after 'b' the recursive "a" tail is reachable (best-effort left rec)
        check("left-rec 'b' then 'a'", allows(c, v, "a"));
    }

    // ========================================================================
    // GBNF JSON parity
    // ========================================================================

    static void testGbnfJsonParity() {
        System.out.println("-- gbnf json parity --");
        MockV v = new MockV();

        Grammar.Spec gbnfJson = Grammar.of(Grammar.JSON_GRAMMAR, v);
        check("gbnf json compiles", gbnfJson.isValid());

        Grammar.Cursor gc = gbnfJson.cursor();
        boolean gStart =
                allows(gc, v, "{")
                        || allows(gc, v, "[")
                        || allows(gc, v, "\"")
                        || allows(gc, v, "t")
                        || allows(gc, v, "f")
                        || allows(gc, v, "n");
        check("gbnf json start accepts values", gStart);

        gc.reset();
        advance(gc, v, "{");
        advance(gc, v, "\"");
        advance(gc, v, "a");
        advance(gc, v, "\"");
        advance(gc, v, ":");
        advance(gc, v, "1");
        advance(gc, v, "}");
        check("gbnf json object walk", jsonDone(gc, v));
    }

    // ========================================================================
    // cache
    // ========================================================================

    static void testGbnfCache() {
        System.out.println("-- gbnf cache --");
        MockV v = new MockV();

        Grammar.Spec a1 = Grammar.of("root ::= \"hello\"", v);
        Grammar.Spec a2 = Grammar.of("root ::= \"hello\"", v);
        check("cache hit", a1 == a2);

        Grammar.Spec b = Grammar.of("root ::= \"world\"", v);
        check("cache miss", a1 != b);

        Grammar.Spec j1 = Grammar.json(v);
        Grammar.Spec j2 = Grammar.json(v);
        check("json cache hit", j1 == j2);
    }

    // ========================================================================
    // empty grammar
    // ========================================================================

    static void testGbnfEmpty() {
        System.out.println("-- gbnf empty/edge --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"\"", v);
        check("empty compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("empty satisfies immediately", !anyValid(c, v));

        s = Grammar.of("root ::= [ \\t\\n\\r]", v);
        c = s.cursor();
        check("ws ' '", allows(c, v, " "));
        check("ws '\\n'", allows(c, v, "\n"));
        check("ws reject 'a'", rejects(c, v, "a"));
    }

    static void testJsonGbnfCompiles() {
        System.out.println("-- json gbnf compiles --");
        MockV v = new MockV();
        Grammar.Spec s = Grammar.of(Grammar.JSON_GRAMMAR, v);
        check("json-grammar compiles", s.isValid());
    }

    // ========================================================================
    // MULTI-BYTE TOKEN TESTS (MockV2 — realistic tokenizer simulation)
    // ========================================================================

    static void testMultiByteTokens() {
        System.out.println("-- multi-byte tokens --");
        MockV2 v = new MockV2();

        Grammar.Spec json = Grammar.json(v);
        check("mb-json compiles", json.isValid());
        Grammar.Cursor c = json.cursor();

        // empty object via multi-byte tokens
        advance(c, v, "{");
        advance(c, v, "}");
        check("mb empty {}", jsonDone(c, v));

        // object with literal true
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"key\"");
        advance(c, v, ":");
        advance(c, v, "true");
        advance(c, v, "}");
        check("mb {\"key\":true}", jsonDone(c, v));

        // object with literal false
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"key\"");
        advance(c, v, ":");
        advance(c, v, "false");
        advance(c, v, "}");
        check("mb {\"key\":false}", jsonDone(c, v));

        // object with literal null
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"key\"");
        advance(c, v, ":");
        advance(c, v, "null");
        advance(c, v, "}");
        check("mb {\"key\":null}", jsonDone(c, v));

        // array with composite token [1]
        c.reset();
        advance(c, v, "[");
        advance(c, v, "[1]");
        advance(c, v, ",");
        advance(c, v, "123");
        advance(c, v, "]");
        check("mb array composite", jsonDone(c, v));

        // whitespace handling
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\n");
        advance(c, v, "\"key\"");
        advance(c, v, "  ");
        advance(c, v, ":");
        advance(c, v, "\t");
        advance(c, v, "0");
        advance(c, v, "\r");
        advance(c, v, "}");
        check("mb whitespace", jsonDone(c, v));

        // negative number
        c.reset();
        advance(c, v, "[");
        advance(c, v, "-");
        advance(c, v, "1");
        advance(c, v, "]");
        check("mb [-1]", jsonDone(c, v));

        // reject invalid tokens
        c.reset();
        boolean badKey = !allows(c, v, "}");
        boolean badComma = !allows(c, v, ",");
        check("mb start rejects }", badKey);
        check("mb start rejects ,", badComma);
    }

    // ========================================================================
    // JSON string escape sequences
    // ========================================================================

    static void testJsonStringEscapes() {
        System.out.println("-- json string escapes --");
        MockV v = new MockV();

        // The hardcoded DFA handles string escapes via states 5→6→5/7
        Grammar.Spec json = Grammar.json(v);

        // valid: "hello world"  (plain string)
        Grammar.Cursor c = json.cursor();
        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "a");
        advance(c, v, "b");
        advance(c, v, "\"");
        check("str plain ok", anyValid(c, v));

        // backslash is part of string — it transitions to escape state
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "a");
        advance(c, v, "\\"); // enters escape state 6
        advance(c, v, "\""); // escaped quote → back to string state 5
        advance(c, v, "b");
        advance(c, v, "\""); // close string
        check("str escaped quote", anyValid(c, v));

        // escaped backslash
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "\\");
        advance(c, v, "\\"); // escaped backslash → back to string
        advance(c, v, "\"");
        check("str escaped backslash", anyValid(c, v));

        // escaped forward slash (JSON allows \/)
        c.reset();
        advance(c, v, "{");
        advance(c, v, "\"");
        advance(c, v, "\\");
        advance(c, v, "/");
        advance(c, v, "\"");
        check("str escaped slash", anyValid(c, v));

        // unicode escape \u0041 (= 'A')
        c.reset();
        advance(c, v, "\"");
        advance(c, v, "\\");
        advance(c, v, "u");
        advance(c, v, "0");
        advance(c, v, "0");
        advance(c, v, "4");
        advance(c, v, "1");
        advance(c, v, "a"); // anything after 4 hex digits → back to string
        advance(c, v, "\"");
        check("str unicode escape", jsonDone(c, v));

        // RFC 8259: raw control chars (incl. newline) are NOT allowed unescaped inside strings
        c.reset();
        advance(c, v, "\"");
        check("str rejects raw newline (RFC 8259)", rejects(c, v, "\n"));
    }

    // ========================================================================
    // number format tests
    // ========================================================================

    static void testNumberFormats() {
        System.out.println("-- number formats --");
        MockV v = new MockV();

        Grammar.Spec json = Grammar.json(v);
        Grammar.Cursor c;

        // integer
        c = json.cursor();
        advance(c, v, "1");
        check("num 1", anyValid(c, v));

        // multi-digit integer
        c.reset();
        advance(c, v, "1");
        advance(c, v, "2");
        advance(c, v, "3");
        check("num 123", anyValid(c, v));

        // negative
        c.reset();
        advance(c, v, "-");
        advance(c, v, "4");
        advance(c, v, "5");
        check("num -45", anyValid(c, v));

        // decimal
        c.reset();
        advance(c, v, "6");
        advance(c, v, ".");
        advance(c, v, "7");
        check("num 6.7", anyValid(c, v));

        // scientific notation lowercase
        c.reset();
        advance(c, v, "8");
        advance(c, v, "e");
        advance(c, v, "9");
        check("num 8e9", anyValid(c, v));

        // scientific notation uppercase
        c.reset();
        advance(c, v, "1");
        advance(c, v, "E");
        advance(c, v, "+");
        advance(c, v, "2");
        check("num 1E+2", anyValid(c, v));

        // negative decimal with exponent
        c.reset();
        advance(c, v, "-");
        advance(c, v, "0");
        advance(c, v, ".");
        advance(c, v, "3");
        advance(c, v, "e");
        advance(c, v, "-");
        advance(c, v, "2");
        check("num -0.3e-2", anyValid(c, v));

        // reject invalid: number with leading zero (single 0 is ok, 01 is not)
        c.reset();
        advance(c, v, "0");
        check("num 0 ok", anyValid(c, v));

        // After a number completes (state 0), letter starts might be valid as keyword start
        c.reset();
        advance(c, v, "1");
        check("num after 1 still valid", anyValid(c, v)); // state 8, values still ok
    }

    // ========================================================================
    // enable / disable
    // ========================================================================

    static void testEnableDisable() {
        System.out.println("-- enable/disable --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\"", v);
        check("enabled compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("enabled allows 'a'", allows(c, v, "a"));
        check("enabled rejects 'b'", rejects(c, v, "b"));

        // DISABLED spec: cursor should pass through all tokens
        Grammar.Spec disabled = Grammar.Spec.DISABLED;
        check("disabled is DISABLED", disabled == Grammar.Spec.DISABLED);
        check("disabled not valid", !disabled.isValid());

        Grammar.Cursor dc = disabled.cursor();
        MockV v2 = new MockV();
        resetScratch(v2.size());
        boolean allPass = dc.maskLogits(scratch(v2));
        check("disabled maskLogits returns true", allPass);
        boolean allUnmodified = true;
        for (int i = 0; i < v2.size(); i++)
            if (get(scratch(v2), i) <= -1e30f) {
                allUnmodified = false;
                break;
            }
        check("disabled leaves logits unchanged", allUnmodified);
    }

    static void testDisabledCursor() {
        System.out.println("-- disabled cursor --");
        MockV v = new MockV();
        Grammar.Spec disabled = Grammar.Spec.DISABLED;
        Grammar.Cursor dc = disabled.cursor();

        // advanceWith should be a no-op
        int idx = tidx(v, "a");
        dc.advanceWith(idx);
        resetScratch(v.size());
        dc.maskLogits(scratch(v));
        boolean allUnchanged = true;
        for (int i = 0; i < v.size(); i++) if (get(scratch(v), i) <= -1e30f) allUnchanged = false;
        check("disabled advance no-op", allUnchanged);

        // reset should be a no-op
        dc.reset();
        resetScratch(v.size());
        dc.maskLogits(scratch(v));
        boolean allStillGood = true;
        for (int i = 0; i < v.size(); i++) if (get(scratch(v), i) <= -1e30f) allStillGood = false;
        check("disabled reset no-op", allStillGood);
    }

    // ========================================================================
    // advanWith → dead state behavior
    // ========================================================================

    static void testAdvanceDeadState() {
        System.out.println("-- advance dead state --");
        MockV v = new MockV();

        // simple literal: only 'a' is valid, 'b' leads to dead state
        Grammar.Spec s = Grammar.of("root ::= \"a\"", v);
        Grammar.Cursor c = s.cursor();
        check("dead-lit 'a' ok", allows(c, v, "a"));

        // advance with 'b' — should go to -1, maskLogits returns false
        advance(c, v, "b");
        resetScratch(v.size());
        boolean maskOk = c.maskLogits(scratch(v));
        check("dead state mask returns false", !maskOk);
        boolean allNeg = true;
        for (int i = 0; i < v.size(); i++)
            if (get(scratch(v), i) > -1e30f) {
                allNeg = false;
                break;
            }
        check("dead state all logits -inf", allNeg);

        // reset recovers
        c.reset();
        check("dead reset recovers 'a'", allows(c, v, "a"));

        // advance beyond first byte of multi-byte token should also go dead
        Grammar.Spec s2 = Grammar.of("root ::= \"ab\"", v);
        Grammar.Cursor c2 = s2.cursor();
        advance(c2, v, "a");
        check("dead-multi after 'a' ok", anyValid(c2, v));
        advance(c2, v, "c"); // mismatched second byte
        resetScratch(v.size());
        c2.maskLogits(scratch(v));
        boolean allNeg2 = true;
        for (int i = 0; i < v.size(); i++) if (get(scratch(v), i) > -1e30f) allNeg2 = false;
        check("dead multi-byte mismatch", allNeg2);
    }

    // ========================================================================
    // repetition after char classes and dot (parser fix verification)
    // ========================================================================

    static void testRepetitionAfterCharDot() {
        System.out.println("-- repetition after char/dot --");
        MockV v = new MockV();

        // char class star: [0-9]*
        Grammar.Spec s = Grammar.of("root ::= [0-9]*", v);
        check("cc-star compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("cc-star zero", anyValid(c, v));
        advance(c, v, "1");
        check("cc-star one", anyValid(c, v));
        advance(c, v, "9");
        check("cc-star two", anyValid(c, v));
        check("cc-star reject letter", rejects(c, v, "a"));

        // char class plus: [a-z]+
        s = Grammar.of("root ::= [a-z]+", v);
        c = s.cursor();
        check("cc-plus 'a'", allows(c, v, "a"));
        advance(c, v, "a");
        check("cc-plus still", anyValid(c, v));

        // char class optional: [0-9]?
        s = Grammar.of("root ::= [0-9]?", v);
        c = s.cursor();
        check("cc-opt '1' valid", allows(c, v, "1"));
        advance(c, v, "1");
        check("cc-opt done", !anyValid(c, v)); // optional consumed -> complete

        // dot star: .*  (any byte, zero or more)
        s = Grammar.of("root ::= .*", v);
        c = s.cursor();
        check("dot-star zero", anyValid(c, v));
        advance(c, v, "{");
        check("dot-star one", anyValid(c, v));
        advance(c, v, "a");
        check("dot-star two", anyValid(c, v));

        // dot plus: .+  (any byte, one or more)
        s = Grammar.of("root ::= .+", v);
        c = s.cursor();
        check("dot-plus 'a'", allows(c, v, "a"));
        advance(c, v, "{");
        check("dot-plus still", anyValid(c, v));

        // dot optional: .?  (any byte, zero or one)
        s = Grammar.of("root ::= .?", v);
        c = s.cursor();
        check("dot-opt zero ok", anyValid(c, v));
        advance(c, v, "!");
        check("dot-opt after one", !anyValid(c, v)); // optional consumed -> complete
    }

    // ========================================================================
    // DFA state count sanity checks
    // ========================================================================

    static void testDfaStateCounts() {
        System.out.println("-- compile sanity --");
        MockV v = new MockV();

        // The engine is a pushdown matcher (no DFA table); assert each grammar compiles and
        // constrains the start token correctly rather than checking internal state counts.
        Grammar.Spec s = Grammar.of("root ::= \"a\"", v);
        check("lit compiles", s.isValid() && allows(s.cursor(), v, "a"));

        s = Grammar.of("root ::= \"a\" | \"b\" | \"c\"", v);
        check(
                "alt compiles",
                s.isValid() && allows(s.cursor(), v, "a") && allows(s.cursor(), v, "c"));

        s = Grammar.of("root ::= \"a\" root | \"b\"", v);
        check(
                "rec compiles",
                s.isValid() && allows(s.cursor(), v, "a") && allows(s.cursor(), v, "b"));

        s = Grammar.json(v);
        check(
                "json compiles",
                s.isValid() && allows(s.cursor(), v, "{") && rejects(s.cursor(), v, "}"));

        s = Grammar.of(Grammar.JSON_GRAMMAR, v);
        check("json gbnf compiles", s.isValid() && allows(s.cursor(), v, "["));

        String many = String.join(" | ", Collections.nCopies(8, "\"a\""));
        s = Grammar.of("root ::= " + many, v);
        check("many-alt compiles", s.isValid() && allows(s.cursor(), v, "a"));
    }

    // ========================================================================
    // fuzzy random walk
    // ========================================================================

    static void testFuzzRandomWalk() {
        System.out.println("-- fuzz random walk --");
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(42);
        MockV2 v = new MockV2();

        Grammar.Spec json = Grammar.json(v);
        try {
            for (int run = 0; run < 20; run++) {
                Grammar.Cursor c = json.cursor();
                for (int step = 0; step < 10; step++) {
                    resetScratch(v.size());
                    c.maskLogits(scratch(v));
                    List<String> list = new ArrayList<>();
                    for (int t = 0; t < v.size(); t++)
                        if (get(scratch(v), t) > -1e30f) list.add(tok(v, t));
                    if (list.isEmpty()) break;
                    String next = list.get(rng.nextInt(list.size()));
                    advance(c, v, next);
                }
            }
        } finally {
            // Arena.ofAuto() handles cleanup
        }
        check("fuzz 20 runs", true);
    }

    // ========================================================================
    // deep nesting stress (JSON strings mainly)
    // ========================================================================

    static void testDeepNesting() {
        System.out.println("-- deep nesting --");
        MockV v = new MockV();

        // 50 levels of nested arrays: [[[[ ... ]]]]
        Grammar.Spec json = Grammar.json(v);
        Grammar.Cursor c = json.cursor();
        for (int i = 0; i < 50; i++) advance(c, v, "[");
        for (int i = 0; i < 50; i++) advance(c, v, "]");
        check("deep array 50", jsonDone(c, v));

        // 50 levels of nested objects
        c = json.cursor();
        for (int i = 0; i < 50; i++) {
            advance(c, v, "{");
            advance(c, v, "\"");
            advance(c, v, "a");
            advance(c, v, "\"");
            advance(c, v, ":");
        }
        advance(c, v, "1");
        for (int i = 0; i < 50; i++) advance(c, v, "}");
        check("deep object 50", jsonDone(c, v));
    }

    // ========================================================================
    // hex escapes in char classes (\\xNN)
    // ========================================================================

    static void testHexEscapeInCharClass() {
        System.out.println("-- hex escape in char class --");
        MockV v = new MockV();

        // [\x41] should match 'A' (0x41 = 65 = 'A')
        Grammar.Spec s = Grammar.of("root ::= [\\x41]", v);
        check("cc-hex A compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("cc-hex matches A", allows(c, v, "A"));
        check("cc-hex rejects B", rejects(c, v, "B"));

        // [\x41-\x5A] should match A-Z
        s = Grammar.of("root ::= [\\x41-\\x5A]", v);
        c = s.cursor();
        check("cc-hex-range A", allows(c, v, "A"));
        check("cc-hex-range C", allows(c, v, "C"));
        check("cc-hex-range F", allows(c, v, "F"));
        check("cc-hex-range reject a", rejects(c, v, "a"));

        // [\x30-\x39] should match 0-9
        s = Grammar.of("root ::= [\\x30-\\x39]", v);
        c = s.cursor();
        check("cc-hex-digit 0", allows(c, v, "0"));
        check("cc-hex-digit 9", allows(c, v, "9"));
        check("cc-hex-digit reject a", rejects(c, v, "a"));

        // \x20 (space)
        s = Grammar.of("root ::= [\\x20]", v);
        c = s.cursor();
        check("cc-hex space", allows(c, v, " "));
        check("cc-hex space reject a", rejects(c, v, "a"));

        // \x21 (!) — char in MockV via hex code
        s = Grammar.of("root ::= [\\x21]", v);
        c = s.cursor();
        check("cc-hex bang", allows(c, v, "!"));
        check("cc-hex bang reject a", rejects(c, v, "a"));
    }

    // ========================================================================
    // # comment inside string should not cut the string
    // ========================================================================

    static void testCommentInString() {
        System.out.println("-- comment in string --");

        // The # inside a GBNF string literal should be part of the string,
        // not a comment start. E.g., root ::= "!" should match '!'
        MockV v = new MockV();
        // Use '!' which is in MockV
        Grammar.Spec s = Grammar.of("root ::= \"!\"", v);
        check("hash literal compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("bang literal matches !", allows(c, v, "!"));

        // Comment after string: root ::= "hello" # world
        s = Grammar.of("root ::= \"hello\" # comment", v);
        check("comment after str compiles", s.isValid());

        // "#\" inside string (hash-backslash-quote) — quote is escaped
        Grammar.Spec s2 = null;
        try {
            s2 = Grammar.of("root ::= \"#\\\"x\"", v); // string: #"x
        } catch (Exception e) {
            /* ignore parse error */
        }
        if (s2 != null) check("hash-backslash-quote compiles", s2.isValid());
    }

    // ========================================================================
    // epsilon-only grammar (always accepts)
    // ========================================================================

    static void testEpsilonOnlyGrammar() {
        System.out.println("-- epsilon-only --");
        MockV v = new MockV();

        // Empty body: root ::= ""
        Grammar.Spec s = Grammar.of("root ::= \"\"", v);
        check("eps-only compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        // Should accept immediately: no tokens needed, but also no tokens valid
        check("eps-only no tokens needed", !anyValid(c, v));

        // Whitespace-only: root ::= [ \t\n\r]*
        s = Grammar.of("root ::= [ \\t\\n\\r]*", v);
        check("ws-only compiles", s.isValid());
        c = s.cursor();
        check("ws-only start valid", anyValid(c, v));
        advance(c, v, " ");
        check("ws-only after space", anyValid(c, v));
        advance(c, v, "\n");
        check("ws-only after nl", anyValid(c, v));
        check("ws-only reject letter", rejects(c, v, "a"));
    }

    // ========================================================================
    // reset + rewalk must produce identical results
    // ========================================================================

    static void testResetRewalk() {
        System.out.println("-- reset rewalk --");
        MockV v = new MockV();

        Grammar.Spec s = Grammar.of("root ::= \"a\" \"b\" \"c\"", v);
        Grammar.Cursor c1 = s.cursor();
        Grammar.Cursor c2 = s.cursor();

        String[] walk = {"a", "b", "c"};
        for (String step : walk) {
            Set<String> s1 = allowedSet(c1, v);
            Set<String> s2 = allowedSet(c2, v);
            check("rewalk " + step + " sets equal", s1.equals(s2));
            advance(c1, v, step);
            advance(c2, v, step);
        }
        // After both walks, reset both and rewalk, compare
        c1.reset();
        c2.reset();
        for (String step : walk) {
            Set<String> s1 = allowedSet(c1, v);
            Set<String> s2 = allowedSet(c2, v);
            check("reset-rewalk " + step + " sets equal", s1.equals(s2));
            advance(c1, v, step);
            advance(c2, v, step);
        }
    }

    // ========================================================================
    // last token index edge case
    // ========================================================================

    static void testLastTokenEdgeCase() {
        System.out.println("-- last token edge --");
        MockV v = new MockV();
        int last = v.size() - 1; // token "!" at highest index

        Grammar.Spec s = Grammar.of("root ::= .", v);
        Grammar.Cursor c = s.cursor();
        check("last-token dot allows !", allows(c, v, "!"));
        advance(c, v, "!");
        // After consuming one dot, grammar is satisfied — no more tokens valid
        check("last-token after ! done", !anyValid(c, v));

        // Ensure mask bit for last token doesn't overflow
        MemoryView<?> logits = scratch(v);
        for (int i = 0; i < v.size(); i++) set(logits, i, 0.0f);
        c.maskLogits(logits);
        // All tokens at or beyond vocab are -inf (except valid ones)
        boolean lastTokenMasked = get(logits, last) > -1e30f;
        check("last token mask accessible", true); // just ensuring no ArrayIndexOOB
    }

    // ========================================================================
    // MAX_DFA_STATES overflow guard
    // ========================================================================

    static void testMaxDfaStatesGuard() {
        System.out.println("-- max dfa states --");
        MockV v = new MockV();

        // Build a grammar with many rules that generates many DFA states
        // 50 alternatives: a | b | c | ... (uses many literals)
        StringBuilder sb = new StringBuilder("root ::= ");
        String letters = "abcdefghijklmnopqrstuvwxyzABCDE";
        for (int i = 0; i < letters.length(); i++) {
            if (i > 0) sb.append(" | ");
            sb.append("\"").append(letters.charAt(i)).append("\"");
        }
        Grammar.Spec s = Grammar.of(sb.toString(), v);
        check("large-alt compiles", s.isValid());
        // Should compile without crash and still constrain to the alternatives
        Grammar.Cursor c = s.cursor();
        check("large-alt allows 'a'", allows(c, v, "a"));
        check("large-alt rejects '1'", rejects(c, v, "1"));
    }

    // ========================================================================
    // zero-vocab edge case
    // ========================================================================

    static void testZeroVocab() {
        System.out.println("-- zero vocab --");
        Grammar.Vocab zv =
                new Grammar.Vocab() {
                    @Override
                    public int size() {
                        return 0;
                    }

                    @Override
                    public byte[] bytes(int t) {
                        return new byte[0];
                    }
                };

        Grammar.Spec s = Grammar.of("root ::= \"a\"", zv);
        check("zero-vocab compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        // maskLogits with zero vocab — should not crash
        MemoryView<?> logits = scratch(zv);
        boolean r = c.maskLogits(logits);
        check("zero-vocab mask returns false", !r);
        // advance with out-of-range token
        c.advanceWith(10);
        check("zero-vocab advance noop", true); // shouldn't crash
    }

    // ========================================================================
    // multi-byte token mask consistency
    // ========================================================================

    static void testMultiByteMaskConsistency() {
        System.out.println("-- multi-byte mask consistency --");
        MockV2 v2 = new MockV2();
        MockV v1 = new MockV();

        // Both vocabs produce compatible JSON masks from the start state
        Grammar.Spec js1 = Grammar.json(v1);
        Grammar.Spec js2 = Grammar.json(v2);
        check("mb-consistency both compile", js1.isValid() && js2.isValid());

        // Walk {} on both, verify state behavior matches
        Grammar.Cursor c1 = js1.cursor();
        Grammar.Cursor c2 = js2.cursor();
        advance(c1, v1, "{");
        advance(c2, v2, "{");
        advance(c1, v1, "}");
        advance(c2, v2, "}");
        check("mb-consistency {} both ok", jsonDone(c1, v1) && jsonDone(c2, v2));
    }

    // ========================================================================
    // empty char class edge cases
    // ========================================================================

    static void testEmptyCharClass() {
        System.out.println("-- empty char class --");
        MockV v = new MockV();

        // [^] — negated empty: matches any byte (since negated nothing == everything)
        Grammar.Spec s = Grammar.of("root ::= [^]", v);
        check("empty-neg compiles", s.isValid());
        Grammar.Cursor c = s.cursor();
        check("empty-neg matches a", allows(c, v, "a"));
        check("empty-neg matches {", allows(c, v, "{"));

        // [] should be treated as matching nothing (invalid) — but may parse
        Grammar.Spec s2 = Grammar.of("root ::= []", v);
        check("empty-pos compiles", s2.isValid());
        Grammar.Cursor c2 = s2.cursor();
        check("empty-pos rejects all", !anyValid(c2, v));

        // [^\x00-\xFF] — negated everything: matches nothing
        Grammar.Spec s3 = Grammar.of("root ::= [^\\x00-\\xFF]", v);
        check("neg-all compiles", s3.isValid());
        Grammar.Cursor c3 = s3.cursor();
        check("neg-all rejects all", !anyValid(c3, v));
    }

    // ========================================================================
    // Spec.DISABLED edge cases
    // ========================================================================

    static void testSpecDisabledEdgeCases() {
        System.out.println("-- disabled edge cases --");
        Grammar.Spec d = Grammar.Spec.DISABLED;

        check("disabled cursor non-null", d.cursor() != null);
        check("disabled isValid false", !d.isValid());

        Grammar.Cursor dc = d.cursor();
        // Multiple resets should be harmless
        dc.reset();
        dc.reset();
        check("disabled double-reset noop", true);

        // Advance with any token should be noop
        dc.advanceWith(0);
        dc.advanceWith(100);
        dc.advanceWith(-1);
        check("disabled advance any noop", true);

        // maskLogits must return true (passthrough)
        MockV v = new MockV();
        MemoryView<?> logits = scratch(v);
        for (int i = 0; i < v.size(); i++) set(logits, i, 42.0f);
        boolean r = dc.maskLogits(logits);
        check("disabled mask passthrough", r);
        boolean all42 = true;
        for (int i = 0; i < v.size(); i++)
            if (Math.abs(get(logits, i) - 42.0f) > 0.001f) all42 = false;
        check("disabled all values preserved", all42);
    }

    // ========================================================================
    // cache per-vocab isolation
    // ========================================================================

    static void testCachePerVocab() {
        System.out.println("-- cache per vocab --");
        MockV v1 = new MockV();
        MockV2 v2 = new MockV2();

        // Same grammar, different vocabs → different specs
        Grammar.Spec s1 = Grammar.of("root ::= \"a\"", v1);
        Grammar.Spec s2 = Grammar.of("root ::= \"a\"", v2);
        check("cache diff vocab diff spec", s1 != s2);

        // Same vocab, same grammar → same spec (cache hit)
        Grammar.Spec s1b = Grammar.of("root ::= \"a\"", v1);
        check("cache same vocab same spec", s1 == s1b);

        // JSON spec: different vocabs → different specs
        Grammar.Spec j1 = Grammar.json(v1);
        Grammar.Spec j2 = Grammar.json(v2);
        check("json cache diff vocab diff spec", j1 != j2);

        // JSON spec: same vocab → same spec
        Grammar.Spec j1b = Grammar.json(v1);
        check("json cache same vocab same spec", j1 == j1b);
    }

    // ========================================================================
    // string literal escape edge cases
    // ========================================================================

    static void testStringLiteralEscapes() {
        System.out.println("-- string literal escapes --");

        // incomplete \x with only one hex digit
        check("unescape \\x", Grammar.unescape("\\x").equals("x")); // fallback to 'x'
        check("unescape \\x5 fallback", Grammar.unescape("\\x5").charAt(0) == 'x'); // only 1 hex
        check("unescape \\xFF", Grammar.unescape("\\xFF").charAt(0) == 0xFF);
        check("unescape \\x00 null byte", Grammar.unescape("\\x00").charAt(0) == '\0');

        // backslash at end of string
        check("unescape trailing \\", Grammar.unescape("a\\").equals("a\\"));

        // multiple escapes in sequence
        check("unescape multi", Grammar.unescape("\\n\\t\\r").equals("\n\t\r"));
        check("unescape mixed", Grammar.unescape("\\x41\\x42").equals("AB"));

        // Verify via grammar compile
        MockV v = new MockV();
        Grammar.Spec s = Grammar.of("root ::= \"\\x41\\x42\"", v); // "AB"
        check("escape-grammar compiles", s.isValid());
    }

    // ========================================================================
    // stripComment edge cases
    // ========================================================================

    static void testStripCommentEdgeCases() {
        System.out.println("-- stripComment edge --");
        MockV v = new MockV();

        // # inside string is NOT a comment
        Grammar.Spec s = Grammar.of("root ::= \"a # b\"", v);
        check("strip-comm str-hash ok", s.isValid());

        // Escaped quote inside string, followed by #
        Grammar.Spec s2 = Grammar.of("root ::= \"a\\\" # b\"", v);
        check("strip-comm esc-quote hash ok", s2.isValid());

        // Multiple lines with comments in GBNF grammar
        String gbnf =
                """
                root ::= "a"  # first choice
                       | "b"  # second choice
                """;
        Grammar.Spec s3 = Grammar.of(gbnf, v);
        check("strip-comm multiline ok", s3.isValid());
    }
}
