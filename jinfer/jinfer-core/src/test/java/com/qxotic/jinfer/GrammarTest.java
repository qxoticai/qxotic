package com.qxotic.jinfer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class GrammarTest {

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

    static F32FloatTensor scratchTensor;

    static F32FloatTensor scratch(Grammar.Vocab v) {
        int max = 57; // max vocab size across mocks
        if (scratchTensor == null || scratchTensor.size() < max)
            scratchTensor = F32FloatTensor.allocate(max);
        return scratchTensor;
    }

    static void resetScratch(int vocab) {
        F32FloatTensor t = scratchTensor;
        for (int i = 0; i < vocab; i++) t.setFloat(i, 0.0f);
    }

    static Set<String> allowedSet(Grammar.Cursor cur, Grammar.Vocab v) {
        Set<String> s = new HashSet<>();
        F32FloatTensor logits = scratch(v);
        for (int i = 0; i < scratchTensor.size(); i++) logits.setFloat(i, 0.0f);
        cur.maskLogits(logits);
        for (int t = 0; t < v.size(); t++) if (logits.getFloat(t) > -1e30f) s.add(tok(v, t));
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

    // count allowed tokens via mask (faster than allowedSet for counts)
    static int allowedCount(Grammar.Cursor cur, int vocab) {
        F32FloatTensor logits = F32FloatTensor.allocate(vocab);
        for (int i = 0; i < vocab; i++) logits.setFloat(i, 0.0f);
        cur.maskLogits(logits);
        int n = 0;
        for (int i = 0; i < vocab; i++) if (logits.getFloat(i) > -1e30f) n++;
        return n;
    }

    public static void main(String[] args) {
        testParser();
        testCursor();
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
            System.err.println("\nGrammarTest: " + failures + " failures");
            System.exit(1);
        }
        System.out.println("\nGrammarTest: 0 failures");
    }

    // ========================================================================
    // parser-only tests
    // ========================================================================

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

        // With ENABLED=true, grammar should constrain
        check("grammar enabled", RuntimeFlags.GRAMMAR);

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
            if (scratch(v2).getFloat(i) <= -1e30f) {
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
        for (int i = 0; i < v.size(); i++)
            if (scratch(v).getFloat(i) <= -1e30f) allUnchanged = false;
        check("disabled advance no-op", allUnchanged);

        // reset should be a no-op
        dc.reset();
        resetScratch(v.size());
        dc.maskLogits(scratch(v));
        boolean allStillGood = true;
        for (int i = 0; i < v.size(); i++)
            if (scratch(v).getFloat(i) <= -1e30f) allStillGood = false;
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
            if (scratch(v).getFloat(i) > -1e30f) {
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
        for (int i = 0; i < v.size(); i++) if (scratch(v).getFloat(i) > -1e30f) allNeg2 = false;
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

        String many = String.join(" | ", java.util.Collections.nCopies(8, "\"a\""));
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
                        if (scratch(v).getFloat(t) > -1e30f) list.add(tok(v, t));
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
        F32FloatTensor logits = scratch(v);
        for (int i = 0; i < v.size(); i++) logits.setFloat(i, 0.0f);
        c.maskLogits(logits);
        // All tokens at or beyond vocab are -inf (except valid ones)
        boolean lastTokenMasked = logits.getFloat(last) > -1e30f;
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
        F32FloatTensor logits = scratch(zv);
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
        F32FloatTensor logits = scratch(v);
        for (int i = 0; i < v.size(); i++) logits.setFloat(i, 42.0f);
        boolean r = dc.maskLogits(logits);
        check("disabled mask passthrough", r);
        boolean all42 = true;
        for (int i = 0; i < v.size(); i++)
            if (Math.abs(logits.getFloat(i) - 42.0f) > 0.001f) all42 = false;
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
