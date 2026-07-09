package com.qxotic.jinfer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Comprehensive, dependency-free tests for the pushdown grammar engine ({@link Grammar}).
 *
 * <p>The engine is exercised purely through its public API ({@link Grammar.Spec#cursor}, {@link
 * Grammar.Cursor#maskLogits}, {@link Grammar.Cursor#advanceWith}) against mock vocabs.
 *
 * <p>Trick that makes this a real membership oracle: every mock vocab contains one <b>EOS</b> token
 * whose decoded bytes are empty. The engine permits an empty-byte token <i>iff</i> it is in an
 * accepting state, so "EOS is unmasked after walking S" == "S is a complete sentence of the
 * grammar". That lets the helpers below distinguish accepted sentences from mere valid prefixes —
 * the distinction the old (broken) implementation never enforced.
 */
public final class GrammarSpecTest {

    static int failures, checks;

    // ---- mock vocabs -------------------------------------------------------

    /** 256 single-byte tokens (id == byte) + an EOS token (id 256, empty bytes). */
    static final class ByteVocab implements Grammar.Vocab {
        @Override
        public int size() {
            return 257;
        }

        @Override
        public byte[] bytes(int t) {
            return t == 256 ? new byte[0] : new byte[] {(byte) t};
        }
    }

    /** A tokenizer-like vocab whose tokens are multi-byte JSON pieces, plus EOS. */
    static final class MultiVocab implements Grammar.Vocab {
        static final String[] W = {
            "{", "}", "[", "]", ":", ",", "\"", "\\\"", "\\", "/", "true", "false", "null", "0",
            "1", "2", "123", "-", ".", "e", "E", "+", "\n", " ", "  ", "\t", "a", "b", "c", "key",
            "\"key\"", "\":", "1}", "]}", "[1]", "1,2",
        };

        @Override
        public int size() {
            return W.length + 1;
        }

        @Override
        public byte[] bytes(int t) {
            return t == W.length ? new byte[0] : W[t].getBytes(StandardCharsets.UTF_8);
        }
    }

    static final ByteVocab BV = new ByteVocab();

    // ---- membership probe (the oracle) -------------------------------------

    static int eosId(Grammar.Vocab v) {
        for (int i = 0; i < v.size(); i++) if (v.bytes(i).length == 0) return i;
        return -1;
    }

    /**
     * {rejectAt, acceptFlag}: rejectAt = index of first disallowed byte (-1 if none); acceptFlag =
     * 1 if EOS is permitted after consuming all bytes (i.e. the string is a complete sentence).
     */
    static int[] probe(Grammar.Spec spec, Grammar.Vocab v, byte[] bytes) {
        Grammar.Cursor c = spec.cursor();
        F32FloatTensor logits = F32FloatTensor.allocate(v.size());
        int eos = eosId(v);
        for (int i = 0; i < bytes.length; i++) {
            mask(c, logits, v.size());
            int b = bytes[i] & 0xFF;
            if (!allowed(logits, b)) return new int[] {i, 0};
            c.advanceWith(b);
        }
        mask(c, logits, v.size());
        return new int[] {-1, allowed(logits, eos) ? 1 : 0};
    }

    static void mask(Grammar.Cursor c, F32FloatTensor logits, int n) {
        for (int i = 0; i < n; i++) logits.setFloat(i, 0f);
        c.maskLogits(logits);
    }

    static boolean allowed(F32FloatTensor logits, int id) {
        return id >= 0 && logits.getFloat(id) > -1e30f;
    }

    static boolean accepts(Grammar.Spec spec, Grammar.Vocab v, String s) {
        int[] r = probe(spec, v, s.getBytes(StandardCharsets.UTF_8));
        return r[0] == -1 && r[1] == 1;
    }

    static boolean notMember(Grammar.Spec spec, Grammar.Vocab v, String s) {
        return !accepts(spec, v, s);
    }

    static int rejectAt(Grammar.Spec spec, Grammar.Vocab v, String s) {
        return probe(spec, v, s.getBytes(StandardCharsets.UTF_8))[0];
    }

    static boolean validPrefix(Grammar.Spec spec, Grammar.Vocab v, String s) {
        return rejectAt(spec, v, s) == -1;
    }

    // byte-vocab convenience (the common case)
    static Grammar.Spec g(String gbnf) {
        return Grammar.of(gbnf, BV);
    }

    static void acc(String name, Grammar.Spec spec, String s) {
        check(name + " ✓accepts " + show(s), accepts(spec, BV, s));
    }

    static void rej(String name, Grammar.Spec spec, String s) {
        check(name + " ✗rejects " + show(s), notMember(spec, BV, s));
    }

    static void rejAt(String name, Grammar.Spec spec, String s, int idx) {
        check(name + " ✗rejects@" + idx + " " + show(s), rejectAt(spec, BV, s) == idx);
    }

    static String show(String s) {
        return "\"" + s.replace("\n", "\\n").replace("\t", "\\t") + "\"";
    }

    // ---- harness -----------------------------------------------------------

    static void check(String what, boolean ok) {
        checks++;
        if (!ok) {
            failures++;
            System.err.println("FAIL: " + what);
        }
    }

    public static void main(String[] args) {
        testJsonValid();
        testJsonInvalid();
        testJsonPreciseRejection();
        testJsonWhitespace();
        testJsonCompact();
        testJsonStrings();
        testJsonNumbers();
        testJsonUnicodeBytes();
        testGbnfLiterals();
        testGbnfCharClasses();
        testGbnfDot();
        testGbnfAlternation();
        testGbnfGroups();
        testGbnfRepetition();
        testGbnfReferences();
        testGbnfRecursion();
        testGbnfEscapes();
        testGbnfEpsilon();
        testMatcherInvariants();
        testDeadAndReset();
        testDisabledAndEdge();
        testMultiByteVocab();
        testDeepNesting();
        testChoice();
        testSchema();
        testFuzzRoundtrip();

        System.out.println("\nGrammarSpecTest: " + checks + " checks, " + failures + " failures");
        if (failures > 0) System.exit(1);
    }

    // ========================================================================
    // JSON — valid sentences
    // ========================================================================

    static final String[] VALID_JSON = {
        "{}",
        "[]",
        "0",
        "-0",
        "123",
        "-123",
        "3.14",
        "-3.14",
        "1e10",
        "1E10",
        "1e+10",
        "1e-10",
        "-2.5e-3",
        "0.5",
        "true",
        "false",
        "null",
        "\"\"",
        "\"hello\"",
        "{\"a\":1}",
        "{\"a\":1,\"b\":2}",
        "{\"a\":true,\"b\":false,\"c\":null}",
        "[1,2,3]",
        "[true,false,null]",
        "[\"a\",\"b\"]",
        "{\"a\":[1,2,3]}",
        "{\"a\":{\"b\":2}}",
        "[[1],[2,3],[]]",
        "{\"a\":[1,{\"b\":[2,3]},4]}",
        "[[[[]]]]",
        "{\"x\":{\"y\":{\"z\":[]}}}",
        "[true,false,null,1,\"s\",{},[]]",
        "{\"k\":\"\"}",
        "{\"empty\":{}}",
        "{\"arr\":[]}",
        "[{\"a\":1},{\"b\":2}]",
        "[-1,-2.5,1e3]",
        "\"with \\\"quote\\\"\"",
        "\"tab\\there\"",
        "\"slash \\/ bs \\\\\"",
        "\"unicode \\u00e9 end\"",
        "\"\\n\\r\\t\\b\\f\"",
    };

    static void testJsonValid() {
        Grammar.Spec j = Grammar.json(BV);
        for (String s : VALID_JSON) acc("json", j, s);
    }

    // ========================================================================
    // JSON — non-sentences (rejected outright, or incomplete prefixes)
    // ========================================================================

    static final String[] INVALID_JSON = {
        // incomplete prefixes (valid so far, not a sentence)
        "{",
        "[",
        "{\"a\"",
        "{\"a\":",
        "[1",
        "[1,",
        "tru",
        "nul",
        "fals",
        "\"unterminated",
        "1.",
        "1e",
        "-",
        "{\"a\":1,",
        // structural errors
        "{\"a\"1}",
        "{\"a\":}",
        "{\"a\":1,}",
        "{,}",
        "{:}",
        "[,]",
        "[1,]",
        "[1 2]",
        "{\"a\":1\"b\":2}",
        "{a:1}",
        "{'a':1}",
        "{\"a\":1}}",
        "[1,2]]",
        "+1",
        "01",
        "1..2",
        ".5",
        "00",
        "{\"a\":01}",
        "truee",
        "nulll",
        "[1,,2]",
        "{\"a\":1,,\"b\":2}",
        "\"bad\\xescape\"",
        "\"\\q\"",
        "}",
        "]",
        ":",
        ",",
    };

    static void testJsonInvalid() {
        Grammar.Spec j = Grammar.json(BV);
        for (String s : INVALID_JSON) rej("json", j, s);
    }

    // ========================================================================
    // JSON — precise rejection point (the original bug class: missing ':')
    // ========================================================================

    static void testJsonPreciseRejection() {
        Grammar.Spec j = Grammar.json(BV);
        rejAt("json missing-colon", j, "{\"a\"1}", 4); // ...the '1' where ':' was due
        rejAt("json missing-comma", j, "[1 2]", 3); // the second number
        rejAt("json unquoted-key", j, "{a:1}", 1); // 'a'
        rejAt("json leading-plus", j, "+1", 0);
        rejAt("json bad-escape", j, "\"\\q\"", 2); // 'q' after backslash
        rejAt("json trailing-comma-arr", j, "[1,]", 3); // ']'
        rejAt("json double-comma", j, "[1,,2]", 3); // second ','
        rejAt("json colon-start", j, ":", 0);
        // a complete value rejects any trailing structural byte
        rejAt("json extra-brace", j, "{}}", 2);
    }

    // ========================================================================
    // JSON — whitespace is optional and allowed between structural tokens
    // ========================================================================

    static void testJsonWhitespace() {
        Grammar.Spec j = Grammar.json(BV);
        acc("ws", j, "{ }");
        acc("ws", j, "[ ]");
        acc("ws", j, "{ \"a\" : 1 }");
        acc("ws", j, "{\n  \"a\" : 1 ,\n  \"b\" : 2\n}");
        acc("ws", j, "[ 1 , 2 , 3 ]");
        acc("ws", j, "[\t1,\n2\r]");
        acc("ws", j, "{\"a\"\t:\t[ 1 , 2 ]}");
        acc("ws", j, "{}"); // no ws
        // top-level surrounding whitespace is allowed (RFC 8259: JSON-text = ws value ws)
        acc("ws", j, " {}");
        acc("ws", j, "{} ");
        acc("ws", j, "  42  ");
        acc("ws", j, "\n[1,2]\n");
        acc("ws", j, " \t true \r ");
    }

    // ========================================================================
    // compact (minified) JSON — no whitespace permitted anywhere
    // ========================================================================

    static void testJsonCompact() {
        Grammar.Spec j = Grammar.jsonCompact(BV);
        // accepts the same structures as JSON, minified
        for (String s :
                new String[] {
                    "{}",
                    "[]",
                    "{\"a\":1}",
                    "{\"a\":1,\"b\":2}",
                    "[1,2,3]",
                    "[true,false,null]",
                    "{\"a\":[1,{\"b\":2}],\"c\":true}",
                    "{\"nested\":{\"deep\":[1,2,{\"x\":\"y\"}]}}",
                    "\"hi\"",
                    "-3.14e2",
                    "true",
                    "null"
                }) acc("compact", j, s);
        // rejects ANY whitespace — between tokens or at the top level
        rej("compact", j, "{ }");
        rej("compact", j, "[ ]");
        rej("compact", j, "{\"a\": 1}");
        rej("compact", j, "{\"a\" :1}");
        rej("compact", j, "[1, 2]");
        rej("compact", j, "[1 ,2]");
        rej("compact", j, "{\"a\":1, \"b\":2}");
        rej("compact", j, " {}"); // leading
        rej("compact", j, "{} "); // trailing
        rej("compact", j, "{\n\"a\":1}"); // newline
        rej("compact", j, "[\t1]"); // tab
        // precise: the space is the first offending byte
        rejAt("compact", j, "[1, 2]", 3);
        rejAt("compact", j, "{} ", 2);
        rejAt("compact", j, " {}", 0);
        // still rejects structural errors and raw control chars, like strict JSON
        rej("compact", j, "{\"a\"1}");
        rej("compact", j, "[1,]");
        rej("compact", j, "\"raw\nnl\"");
        // and json() (pretty) accepts a spaced form the compact one rejects — sanity contrast
        check(
                "compact vs pretty contrast",
                accepts(Grammar.json(BV), BV, "[1, 2]") && notMember(j, BV, "[1, 2]"));
    }

    // ========================================================================
    // JSON — strings & escapes
    // ========================================================================

    static void testJsonStrings() {
        Grammar.Spec j = Grammar.json(BV);
        acc("str", j, "\"\"");
        acc("str", j, "\"a b c\"");
        acc("str", j, "\"!@#$%^&*()\"");
        acc("str", j, "\"escaped quote: \\\" done\"");
        acc("str", j, "\"backslash \\\\ done\"");
        acc("str", j, "\"forward \\/ slash\"");
        acc("str", j, "\"controls \\b\\f\\n\\r\\t\"");
        acc("str", j, "\"u-escape \\u0041\\uffff\"");
        rej("str", j, "\"bad \\x41\""); // \x not a JSON escape
        rej("str", j, "\"bad \\u00\""); // too few hex digits
        rej("str", j, "\"unterminated");
        rej("str", j, "\"u \\u00gg\""); // non-hex
        // RFC 8259: unescaped control chars (0x00-0x1F) are NOT allowed inside strings
        rej("str", j, "\"raw\nnewline\""); // real 0x0A byte inside the string
        rej("str", j, "\"raw\ttab\""); // real 0x09 byte
        acc("str", j, "\"esc \\n \\t \\r ok\""); // their escaped forms are fine
        // key strings constrain the same way
        acc("str", j, "{\"k\\ney\":1}"); // escaped newline in key
        rej("str", j, "{\"k\ney\":1}"); // raw newline in key rejected
        // the letter F must NOT be excluded by the 0x00-0x1F exclusion (parser hex-range
        // regression)
        acc("str", j, "\"F G H\"");
    }

    // ========================================================================
    // JSON — numbers
    // ========================================================================

    static void testJsonNumbers() {
        Grammar.Spec j = Grammar.json(BV);
        for (String n :
                new String[] {
                    "0",
                    "-0",
                    "1",
                    "-1",
                    "42",
                    "-42",
                    "3.14",
                    "-3.14",
                    "0.0",
                    "1e0",
                    "1e5",
                    "1E5",
                    "1e+5",
                    "1e-5",
                    "1.5e10",
                    "-1.5E-10",
                    "123456789"
                }) acc("num", j, n);
        for (String n :
                new String[] {
                    "01", "00", "1.", ".1", "1e", "1e+", "+1", "1..0", "1.2.3", "--1", "1-2", "0x1",
                    "1.2e", "."
                }) rej("num", j, n);
    }

    // ========================================================================
    // JSON — multi-byte UTF-8 passes through inside strings
    // ========================================================================

    static void testJsonUnicodeBytes() {
        Grammar.Spec j = Grammar.json(BV);
        acc("utf8", j, "\"café\""); // é (2 bytes)
        acc("utf8", j, "\"€\""); // € (3 bytes)
        acc("utf8", j, "\"😀 smile\""); // 😀 (4 bytes)
        acc("utf8", j, "{\"中文\":\"值\"}"); // CJK key+value
    }

    // ========================================================================
    // GBNF — literals
    // ========================================================================

    static void testGbnfLiterals() {
        Grammar.Spec s = g("root ::= \"abc\"");
        acc("lit", s, "abc");
        rej("lit", s, "ab");
        rej("lit", s, "abcd");
        rej("lit", s, "abx");
        rej("lit", s, "");
        rejAt("lit", s, "abx", 2);

        Grammar.Spec one = g("root ::= \"x\"");
        acc("lit1", one, "x");
        rej("lit1", one, "");
        rej("lit1", one, "xx");

        Grammar.Spec seq = g("root ::= \"a\" \"b\" \"c\"");
        acc("seq", seq, "abc");
        rej("seq", seq, "ab");
        rej("seq", seq, "abcd");
    }

    // ========================================================================
    // GBNF — character classes
    // ========================================================================

    static void testGbnfCharClasses() {
        Grammar.Spec lower = g("root ::= [a-z]");
        acc("cc", lower, "a");
        acc("cc", lower, "m");
        acc("cc", lower, "z");
        rej("cc", lower, "A");
        rej("cc", lower, "1");
        rej("cc", lower, "");
        rej("cc", lower, "ab");

        Grammar.Spec alnum = g("root ::= [a-zA-Z0-9]+");
        acc("cc+", alnum, "aZ09");
        acc("cc+", alnum, "X");
        rej("cc+", alnum, "");
        rej("cc+", alnum, "a-b");

        Grammar.Spec neg = g("root ::= [^0-9]");
        acc("ccneg", neg, "a");
        acc("ccneg", neg, " ");
        acc("ccneg", neg, "!");
        rej("ccneg", neg, "0");
        rej("ccneg", neg, "5");

        Grammar.Spec mixed = g("root ::= [a-cx-z]");
        acc("ccmix", mixed, "a");
        acc("ccmix", mixed, "c");
        acc("ccmix", mixed, "y");
        rej("ccmix", mixed, "d");
        rej("ccmix", mixed, "w");

        Grammar.Spec esc = g("root ::= [\\n\\t ]+");
        acc("ccesc", esc, "\n\t ");
        rej("ccesc", esc, "a");

        Grammar.Spec hex = g("root ::= [\\x41-\\x43]");
        acc("cchex", hex, "A");
        acc("cchex", hex, "B");
        acc("cchex", hex, "C");
        rej("cchex", hex, "D");

        // hex ranges must be EXACT — regression for the off-by-one that re-read the range-end
        // token's last hex digit as a spurious extra member (e.g. wrongly admitting ':' or 'F').
        Grammar.Spec dig = g("root ::= [\\x30-\\x39]"); // exactly '0'..'9'
        acc("cchexexact", dig, "0");
        acc("cchexexact", dig, "9");
        rej("cchexexact", dig, ":"); // 0x3A, one past '9' — the bug's victim
        rej("cchexexact", dig, "/"); // 0x2F, one before '0'
        Grammar.Spec ctl = g("root ::= [\\x00-\\x1F]"); // control chars only
        acc("cchexctl", ctl, "\u0005");
        acc("cchexctl", ctl, "\u001f");
        rej("cchexctl", ctl, "F"); // 0x46 — was spuriously admitted
        rej("cchexctl", ctl, " "); // 0x20, one past the range
    }

    // ========================================================================
    // GBNF — dot
    // ========================================================================

    static void testGbnfDot() {
        Grammar.Spec dot = g("root ::= .");
        acc("dot", dot, "a");
        acc("dot", dot, "{");
        acc("dot", dot, "\n");
        rej("dot", dot, "");
        rej("dot", dot, "ab");

        Grammar.Spec mid = g("root ::= \"a\" . \"c\"");
        acc("dotmid", mid, "abc");
        acc("dotmid", mid, "axc");
        acc("dotmid", mid, "a c");
        rej("dotmid", mid, "ac");
        rej("dotmid", mid, "abbc");
    }

    // ========================================================================
    // GBNF — alternation
    // ========================================================================

    static void testGbnfAlternation() {
        Grammar.Spec alt = g("root ::= \"a\" | \"b\" | \"c\"");
        acc("alt", alt, "a");
        acc("alt", alt, "b");
        acc("alt", alt, "c");
        rej("alt", alt, "d");
        rej("alt", alt, "");
        rej("alt", alt, "ab");

        Grammar.Spec words = g("root ::= \"cat\" | \"car\" | \"dog\"");
        acc("altw", words, "cat");
        acc("altw", words, "car");
        acc("altw", words, "dog");
        rej("altw", words, "ca");
        rej("altw", words, "cab");
        rej("altw", words, "do");

        Grammar.Spec seq = g("root ::= \"x\" (\"a\" | \"bb\") \"y\"");
        acc("altseq", seq, "xay");
        acc("altseq", seq, "xbby");
        rej("altseq", seq, "xby");
        rej("altseq", seq, "xy");
    }

    // ========================================================================
    // GBNF — groups
    // ========================================================================

    static void testGbnfGroups() {
        Grammar.Spec nested = g("root ::= \"a\" (\"b\" (\"c\" | \"d\"))");
        acc("grp", nested, "abc");
        acc("grp", nested, "abd");
        rej("grp", nested, "ab");
        rej("grp", nested, "abe");

        Grammar.Spec rep = g("root ::= \"<\" (\"a\" \"b\")* \">\"");
        acc("grprep", rep, "<>");
        acc("grprep", rep, "<ab>");
        acc("grprep", rep, "<abab>");
        rej("grprep", rep, "<a>");
        rej("grprep", rep, "<aba>");
    }

    // ========================================================================
    // GBNF — repetition  *  +  ?
    // ========================================================================

    static void testGbnfRepetition() {
        Grammar.Spec star = g("root ::= \"a\"*");
        acc("star", star, "");
        acc("star", star, "a");
        acc("star", star, "aaaa");
        rej("star", star, "b");
        rej("star", star, "ab");

        Grammar.Spec plus = g("root ::= \"a\"+");
        rej("plus", plus, "");
        acc("plus", plus, "a");
        acc("plus", plus, "aaa");

        Grammar.Spec opt = g("root ::= \"a\"?");
        acc("opt", opt, "");
        acc("opt", opt, "a");
        rej("opt", opt, "aa");

        Grammar.Spec ccstar = g("root ::= [0-9]*");
        acc("ccstar", ccstar, "");
        acc("ccstar", ccstar, "007");
        rej("ccstar", ccstar, "0a");

        Grammar.Spec combo = g("root ::= \"x\" [0-9]+ \"y\"?");
        acc("combo", combo, "x1");
        acc("combo", combo, "x1y");
        acc("combo", combo, "x42");
        rej("combo", combo, "x");
        rej("combo", combo, "xy");
        rej("combo", combo, "x1yy");

        Grammar.Spec grpstar = g("root ::= (\"ab\" | \"cd\")*");
        acc("grpstar", grpstar, "");
        acc("grpstar", grpstar, "ab");
        acc("grpstar", grpstar, "abcdab");
        rej("grpstar", grpstar, "a");
        rej("grpstar", grpstar, "abc");
    }

    // ========================================================================
    // GBNF — references / rule chains
    // ========================================================================

    static void testGbnfReferences() {
        Grammar.Spec chain =
                g(
                        """
                        root ::= a b
                        a ::= "x" | "y"
                        b ::= [0-9]+
                        """);
        acc("ref", chain, "x1");
        acc("ref", chain, "y42");
        rej("ref", chain, "x");
        rej("ref", chain, "z1");

        Grammar.Spec deep =
                g(
                        """
                        root ::= a
                        a ::= b
                        b ::= c
                        c ::= "deep"
                        """);
        acc("refdeep", deep, "deep");
        rej("refdeep", deep, "dee");

        // a rule referenced from multiple sites (the case the old flat-NFA mishandled)
        Grammar.Spec multi =
                g(
                        """
                        root ::= word " " word
                        word ::= [a-z]+
                        """);
        acc("refmulti", multi, "foo bar");
        acc("refmulti", multi, "a b");
        rej("refmulti", multi, "foo  bar");
        rej("refmulti", multi, "foo bar baz");
    }

    // ========================================================================
    // GBNF — recursion (right-recursive: fully supported)
    // ========================================================================

    static void testGbnfRecursion() {
        Grammar.Spec parens = g("root ::= \"(\" root \")\" | \"\"");
        acc("rec", parens, "");
        acc("rec", parens, "()");
        acc("rec", parens, "(())");
        acc("rec", parens, "((((()))))");
        rej("rec", parens, "(");
        rej("rec", parens, "(()");
        rej("rec", parens, "())");
        rej("rec", parens, ")(");

        Grammar.Spec list =
                g(
                        """
                        root  ::= "[" inner "]"
                        inner ::= "" | item ("," item)*
                        item  ::= [0-9]+ | root
                        """);
        acc("reclist", list, "[]");
        acc("reclist", list, "[1]");
        acc("reclist", list, "[1,2,3]");
        acc("reclist", list, "[[]]");
        acc("reclist", list, "[[1],2]");
        acc("reclist", list, "[1,[2,[3]]]");
        rej("reclist", list, "[1,]");
        rej("reclist", list, "[,]");
        rej("reclist", list, "[1");
        rej("reclist", list, "1]");

        Grammar.Spec expr =
                g(
                        """
                        expr ::= term ("+" term)*
                        term ::= [0-9]+
                        """);
        acc("recexpr", expr, "1");
        acc("recexpr", expr, "1+2");
        acc("recexpr", expr, "12+34+56");
        rej("recexpr", expr, "1+");
        rej("recexpr", expr, "+1");
        rej("recexpr", expr, "1++2");
        rej("recexpr", expr, "1+2+");
    }

    // ========================================================================
    // GBNF — escapes in literals
    // ========================================================================

    static void testGbnfEscapes() {
        Grammar.Spec nl = g("root ::= \"a\\nb\"");
        acc("escnl", nl, "a\nb");
        rej("escnl", nl, "ab");
        rej("escnl", nl, "a b");

        Grammar.Spec hex = g("root ::= \"\\x41\\x42\""); // "AB"
        acc("eschex", hex, "AB");
        rej("eschex", hex, "ab");

        Grammar.Spec tab = g("root ::= \"x\\ty\"");
        acc("esctab", tab, "x\ty");
        rej("esctab", tab, "xy");
    }

    // ========================================================================
    // GBNF — epsilon / empty
    // ========================================================================

    static void testGbnfEpsilon() {
        Grammar.Spec empty = g("root ::= \"\"");
        acc("eps", empty, "");
        rej("eps", empty, "a");

        Grammar.Spec wsOnly = g("root ::= [ \\t\\n\\r]*");
        acc("wsonly", wsOnly, "");
        acc("wsonly", wsOnly, "   ");
        acc("wsonly", wsOnly, " \t\n");
        rej("wsonly", wsOnly, "x");

        Grammar.Spec optChain = g("root ::= \"a\"? \"b\"? \"c\"?");
        acc("optchain", optChain, "");
        acc("optchain", optChain, "a");
        acc("optchain", optChain, "ac");
        acc("optchain", optChain, "abc");
        acc("optchain", optChain, "bc");
        rej("optchain", optChain, "ba");
        rej("optchain", optChain, "aa");
    }

    // ========================================================================
    // matcher invariants
    // ========================================================================

    static void testMatcherInvariants() {
        Grammar.Spec j = Grammar.json(BV);

        // determinism: two independent cursors give identical allowed sets along the same walk
        Grammar.Cursor c1 = j.cursor(), c2 = j.cursor();
        F32FloatTensor l1 = F32FloatTensor.allocate(BV.size()),
                l2 = F32FloatTensor.allocate(BV.size());
        boolean same = true;
        for (byte b : "{\"a\":[1,2]}".getBytes(StandardCharsets.UTF_8)) {
            mask(c1, l1, BV.size());
            mask(c2, l2, BV.size());
            for (int i = 0; i < BV.size(); i++)
                if ((l1.getFloat(i) > -1e30f) != (l2.getFloat(i) > -1e30f)) same = false;
            c1.advanceWith(b & 0xFF);
            c2.advanceWith(b & 0xFF);
        }
        check("inv determinism", same);

        // reset restores the initial allowed set
        Grammar.Cursor c = j.cursor();
        List<Boolean> startMask = snapshot(c);
        for (byte b : "{\"a\":1".getBytes(StandardCharsets.UTF_8)) c.advanceWith(b & 0xFF);
        c.reset();
        check("inv reset", snapshot(c).equals(startMask));

        // every byte the mask permits leads to a non-dead cursor (mask&advance agree)
        Grammar.Spec rec = g("root ::= \"(\" root \")\" | \"x\"");
        check("inv mask-advance agree", maskAdvanceAgree(rec, 200, 9L));
        check("inv mask-advance agree json", maskAdvanceAgree(j, 200, 7L));
    }

    static List<Boolean> snapshot(Grammar.Cursor c) {
        F32FloatTensor l = F32FloatTensor.allocate(BV.size());
        mask(c, l, BV.size());
        List<Boolean> out = new ArrayList<>();
        for (int i = 0; i < BV.size(); i++) out.add(l.getFloat(i) > -1e30f);
        return out;
    }

    /**
     * Walk randomly through the grammar; assert that any byte the mask allows can in fact be
     * consumed without the cursor going dead unless it was a completing move.
     */
    static boolean maskAdvanceAgree(Grammar.Spec spec, int steps, long seed) {
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(seed);
        F32FloatTensor l = F32FloatTensor.allocate(BV.size());
        Grammar.Cursor c = spec.cursor();
        int eos = eosId(BV);
        for (int step = 0; step < steps; step++) {
            mask(c, l, BV.size());
            List<Integer> bytes = new ArrayList<>();
            for (int b = 0; b < 256; b++) if (l.getFloat(b) > -1e30f) bytes.add(b);
            boolean eosOk = l.getFloat(eos) > -1e30f;
            if (bytes.isEmpty()) return eosOk; // dead end must be an accepting end
            if (eosOk && rng.nextInt(3) == 0) return true; // optionally stop at an accepting point
            c.advanceWith(bytes.get(rng.nextInt(bytes.size())));
        }
        return true;
    }

    // ========================================================================
    // dead state & reset
    // ========================================================================

    static void testDeadAndReset() {
        Grammar.Spec s = g("root ::= \"abc\"");
        Grammar.Cursor c = s.cursor();
        c.advanceWith('a');
        c.advanceWith('z'); // 'z' is impossible -> dead
        F32FloatTensor l = F32FloatTensor.allocate(BV.size());
        boolean any = c.maskLogits(zero(l));
        check("dead: nothing allowed", !any);
        boolean allNeg = true;
        for (int i = 0; i < BV.size(); i++) if (l.getFloat(i) > -1e30f) allNeg = false;
        check("dead: all -inf", allNeg);
        c.reset();
        check("dead: reset recovers", accepts(s, BV, "abc") && validPrefix(s, BV, "ab"));

        // advancing past the end (token out of range / extra byte) stays safe
        Grammar.Cursor c2 = s.cursor();
        for (byte b : "abc".getBytes(StandardCharsets.UTF_8)) c2.advanceWith(b & 0xFF);
        c2.advanceWith('x'); // beyond a complete sentence -> dead
        check("dead: past-end", !c2.maskLogits(zero(l)));
        c2.advanceWith(99999); // out-of-range token id: no crash
        check("dead: oob-token noop", true);
    }

    static F32FloatTensor zero(F32FloatTensor l) {
        for (int i = 0; i < l.size(); i++) l.setFloat(i, 0f);
        return l;
    }

    // ========================================================================
    // disabled spec & degenerate vocabs
    // ========================================================================

    static void testDisabledAndEdge() {
        Grammar.Spec d = Grammar.Spec.DISABLED;
        check("disabled invalid", !d.isValid());
        Grammar.Cursor dc = d.cursor();
        F32FloatTensor l = F32FloatTensor.allocate(BV.size());
        for (int i = 0; i < BV.size(); i++) l.setFloat(i, 5f);
        check("disabled passthrough true", dc.maskLogits(l));
        boolean untouched = true;
        for (int i = 0; i < BV.size(); i++) if (l.getFloat(i) != 5f) untouched = false;
        check("disabled leaves logits", untouched);
        dc.advanceWith(0);
        dc.advanceWith(-1);
        dc.reset(); // all no-ops, no crash
        check("disabled ops noop", true);

        // empty grammar text -> DISABLED
        check(
                "empty grammar -> disabled",
                Grammar.of("", BV) == Grammar.Spec.DISABLED || !Grammar.of("", BV).isValid());

        // zero-size vocab
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
        Grammar.Spec zs = Grammar.of("root ::= \"a\"", zv);
        check("zero vocab valid", zs.isValid());
        check("zero vocab mask false", !zs.cursor().maskLogits(F32FloatTensor.allocate(1)));

        // spec caching identity
        check(
                "cache hit same grammar",
                Grammar.of("root ::= \"q\"", BV) == Grammar.of("root ::= \"q\"", BV));
        check(
                "cache miss diff grammar",
                Grammar.of("root ::= \"q\"", BV) != Grammar.of("root ::= \"r\"", BV));
        check("json cache hit", Grammar.json(BV) == Grammar.json(BV));
    }

    // ========================================================================
    // multi-byte tokenizer vocab (tokens spanning grammar boundaries)
    // ========================================================================

    static void testMultiByteVocab() {
        MultiVocab v = new MultiVocab();
        Grammar.Spec j = Grammar.json(v);
        check("mb json compiles", j.isValid());

        // tokens are whole pieces; build sentences from byte-correct decompositions
        accV("mb", j, v, "{", "\"key\"", ":", "true", "}"); // {"key":true}
        accV("mb", j, v, "{", "}"); // {}
        accV("mb", j, v, "[", "]"); // []
        accV("mb", j, v, "[", "1,2", "]"); // [1,2] via composite "1,2"
        accV("mb", j, v, "{", "\"key\"", ":", "[1]", "}"); // {"key":[1]} via composite "[1]"
        accV("mb", j, v, "[", "{", "\"key\"", ":", "null", "}", "]"); // [{"key":null}]
        accV("mb", j, v, "true");
        accV("mb", j, v, "-", "123", ".", "123", "e", "-", "1"); // a number across tokens

        // composite closing tokens "1}" and "]}" close structure in one token
        accV("mb", j, v, "{", "\"key\"", ":", "1}"); // {"key":1}  via "1}"
        accV("mb", j, v, "{", "\"key\"", ":", "[", "1", "]}"); // {"key":[1]} via "]}"

        rejV("mb", j, v, "{", "}", "}"); // extra brace
        rejV("mb", j, v, "{", ","); // comma where key expected
        rejV("mb", j, v, "[", ",", "]"); // leading comma
        rejV("mb", j, v, "{", "\"key\"", "\"key\""); // double key, no colon
    }

    /** accepts a token-id sequence (each arg matched to a vocab word) as a complete sentence. */
    static void accV(String name, Grammar.Spec spec, Grammar.Vocab v, String... toks) {
        check(name + " ✓accepts " + String.join("·", toks), acceptsTokens(spec, v, toks));
    }

    static void rejV(String name, Grammar.Spec spec, Grammar.Vocab v, String... toks) {
        check(name + " ✗rejects " + String.join("·", toks), !acceptsTokens(spec, v, toks));
    }

    static boolean acceptsTokens(Grammar.Spec spec, Grammar.Vocab v, String[] toks) {
        Grammar.Cursor c = spec.cursor();
        F32FloatTensor l = F32FloatTensor.allocate(v.size());
        for (String t : toks) {
            int id = tokenId(v, t);
            if (id < 0) return false;
            mask(c, l, v.size());
            if (!(l.getFloat(id) > -1e30f)) return false;
            c.advanceWith(id);
        }
        mask(c, l, v.size());
        return l.getFloat(eosId(v)) > -1e30f;
    }

    static int tokenId(Grammar.Vocab v, String s) {
        byte[] want = s.getBytes(StandardCharsets.UTF_8);
        for (int i = 0; i < v.size(); i++) if (java.util.Arrays.equals(v.bytes(i), want)) return i;
        return -1;
    }

    // ========================================================================
    // deep nesting (stack depth, no overflow / cliff)
    // ========================================================================

    static void testDeepNesting() {
        Grammar.Spec j = Grammar.json(BV);
        for (int depth : new int[] {8, 64, 256}) {
            String s = "[".repeat(depth) + "]".repeat(depth);
            check("deep array " + depth, accepts(j, BV, s));
            check(
                    "deep array unbalanced " + depth,
                    notMember(j, BV, "[".repeat(depth) + "]".repeat(depth - 1)));
        }
        // deep objects
        StringBuilder open = new StringBuilder(), close = new StringBuilder();
        int d = 32;
        for (int i = 0; i < d; i++) {
            open.append("{\"a\":");
            close.append("}");
        }
        check("deep object " + d, accepts(j, BV, open + "1" + close));
    }

    // ========================================================================
    // enum / choice
    // ========================================================================

    static void testChoice() {
        Grammar.Spec yn = Grammar.choice(BV, "yes", "no");
        acc("choice", yn, "yes");
        acc("choice", yn, "no");
        rej("choice", yn, "maybe");
        rej("choice", yn, "ye");
        rej("choice", yn, "yes ");
        rej("choice", yn, "");

        Grammar.Spec cat = Grammar.choice(BV, "positive", "negative", "neutral");
        acc("choice3", cat, "positive");
        acc("choice3", cat, "neutral");
        rej("choice3", cat, "Positive");
        rej("choice3", cat, "neg");
    }

    // ========================================================================
    // JSON Schema -> grammar
    // ========================================================================

    static Map<String, Object> map(Object... kv) {
        LinkedHashMap<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    static Grammar.Spec sc(Map<String, Object> schema) {
        return Grammar.fromSchema(schema, BV);
    }

    static void testSchema() {
        // scalar types
        Grammar.Spec str = sc(map("type", "string"));
        acc("sc-str", str, "\"hi\"");
        acc("sc-str", str, "\"\"");
        rej("sc-str", str, "hi");
        rej("sc-str", str, "42");

        Grammar.Spec integer = sc(map("type", "integer"));
        acc("sc-int", integer, "42");
        acc("sc-int", integer, "-7");
        acc("sc-int", integer, "0");
        rej("sc-int", integer, "4.2");
        rej("sc-int", integer, "\"4\"");
        rej("sc-int", integer, "01");

        Grammar.Spec num = sc(map("type", "number"));
        acc("sc-num", num, "3.14");
        acc("sc-num", num, "-1e5");
        acc("sc-num", num, "0");
        rej("sc-num", num, "abc");
        rej("sc-num", num, "\"3\"");

        Grammar.Spec bool = sc(map("type", "boolean"));
        acc("sc-bool", bool, "true");
        acc("sc-bool", bool, "false");
        rej("sc-bool", bool, "True");
        rej("sc-bool", bool, "1");

        Grammar.Spec nul = sc(map("type", "null"));
        acc("sc-null", nul, "null");
        rej("sc-null", nul, "nil");

        // enum (JSON-encoded literals) and const
        Grammar.Spec col = sc(map("enum", Arrays.asList("red", "green", "blue")));
        acc("sc-enum", col, "\"red\"");
        acc("sc-enum", col, "\"blue\"");
        rej("sc-enum", col, "red");
        rej("sc-enum", col, "\"yellow\"");

        Grammar.Spec enumNum = sc(map("enum", Arrays.asList(1, 2, 3)));
        acc("sc-enumnum", enumNum, "1");
        acc("sc-enumnum", enumNum, "3");
        rej("sc-enumnum", enumNum, "4");

        Grammar.Spec cst = sc(map("const", "fixed"));
        acc("sc-const", cst, "\"fixed\"");
        rej("sc-const", cst, "\"other\"");

        // type unions and anyOf
        Grammar.Spec nullable = sc(map("type", Arrays.asList("string", "null")));
        acc("sc-union", nullable, "\"x\"");
        acc("sc-union", nullable, "null");
        rej("sc-union", nullable, "5");

        Grammar.Spec any =
                sc(map("anyOf", Arrays.asList(map("type", "string"), map("type", "integer"))));
        acc("sc-anyof", any, "\"s\"");
        acc("sc-anyof", any, "42");
        rej("sc-anyof", any, "true");

        // object with required properties (emitted in order)
        Grammar.Spec person =
                sc(
                        map(
                                "type", "object",
                                "properties",
                                        map(
                                                "name",
                                                map("type", "string"),
                                                "age",
                                                map("type", "integer")),
                                "required", Arrays.asList("name", "age")));
        acc("sc-obj", person, "{\"name\":\"Bob\",\"age\":30}");
        acc("sc-obj", person, "{ \"name\" : \"Bob\" , \"age\" : 30 }"); // whitespace
        rej("sc-obj", person, "{\"name\":\"Bob\"}"); // missing age
        rej("sc-obj", person, "{\"name\":30,\"age\":30}"); // name not a string
        rej("sc-obj", person, "{\"age\":30,\"name\":\"Bob\"}"); // wrong order (documented)
        rej("sc-obj", person, "{}");

        // empty / no-properties object
        Grammar.Spec emptyObj = sc(map("type", "object"));
        acc("sc-emptyobj", emptyObj, "{}");
        acc("sc-emptyobj", emptyObj, "{ }");

        // arrays
        Grammar.Spec ints = sc(map("type", "array", "items", map("type", "integer")));
        acc("sc-arr", ints, "[]");
        acc("sc-arr", ints, "[1]");
        acc("sc-arr", ints, "[1,2,3]");
        acc("sc-arr", ints, "[ 1 , 2 ]");
        rej("sc-arr", ints, "[1,\"a\"]");
        rej("sc-arr", ints, "[1,]");

        Grammar.Spec tags = sc(map("type", "array", "items", map("type", "string")));
        acc("sc-arrstr", tags, "[\"a\",\"b\"]");
        rej("sc-arrstr", tags, "[\"a\",1]");

        // nested object + array (recursion through generated rules)
        Grammar.Spec doc =
                sc(
                        map(
                                "type", "object",
                                "properties",
                                        map(
                                                "user",
                                                map(
                                                        "type",
                                                        "object",
                                                        "properties",
                                                        map(
                                                                "id",
                                                                map("type", "integer"),
                                                                "roles",
                                                                map(
                                                                        "type",
                                                                        "array",
                                                                        "items",
                                                                        map("type", "string"))),
                                                        "required",
                                                        Arrays.asList("id", "roles"))),
                                "required", Arrays.asList("user")));
        acc("sc-nested", doc, "{\"user\":{\"id\":5,\"roles\":[\"admin\",\"user\"]}}");
        acc("sc-nested", doc, "{\"user\":{\"id\":5,\"roles\":[]}}");
        rej("sc-nested", doc, "{\"user\":{\"id\":\"5\",\"roles\":[]}}"); // id must be integer
        rej("sc-nested", doc, "{\"user\":{\"roles\":[]}}"); // missing id

        // enum-valued property
        Grammar.Spec rated =
                sc(
                        map(
                                "type", "object",
                                "properties",
                                        map("rating", map("enum", Arrays.asList("good", "bad"))),
                                "required", Arrays.asList("rating")));
        acc("sc-enumprop", rated, "{\"rating\":\"good\"}");
        rej("sc-enumprop", rated, "{\"rating\":\"meh\"}");
        rej("sc-enumprop", rated, "{\"rating\":good}");

        // no constraints -> any JSON
        Grammar.Spec anyJson = sc(map());
        acc("sc-any", anyJson, "{\"a\":[1,2,{\"b\":true}]}");
        acc("sc-any", anyJson, "42");
        acc("sc-any", anyJson, "null");
        rej("sc-any", anyJson, "{\"a\":}");
    }

    // ========================================================================
    // property test: anything the engine GENERATES, it ACCEPTS
    // (catches mask/advance divergence — the failure mode of the old impl)
    // ========================================================================

    static void testFuzzRoundtrip() {
        roundtrip("json", Grammar.json(BV), 64, 11L);
        roundtrip("parens", g("root ::= \"(\" root \")\" | \"x\""), 64, 22L);
        roundtrip(
                "list",
                g(
                        """
                        root  ::= "[" inner "]"
                        inner ::= "" | item ("," item)*
                        item  ::= [0-9] | root
                        """),
                64,
                33L);
        roundtrip("words", g("root ::= ([a-c] | \"--\")*"), 64, 44L);
    }

    static void roundtrip(String name, Grammar.Spec spec, int runs, long seed) {
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(seed);
        F32FloatTensor l = F32FloatTensor.allocate(BV.size());
        int eos = eosId(BV);
        int generated = 0;
        for (int run = 0; run < runs; run++) {
            Grammar.Cursor c = spec.cursor();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            boolean completed = false;
            for (int step = 0; step < 400; step++) {
                mask(c, l, BV.size());
                List<Integer> bytes = new ArrayList<>();
                for (int b = 0; b < 256; b++) if (l.getFloat(b) > -1e30f) bytes.add(b);
                boolean eosOk = l.getFloat(eos) > -1e30f;
                // bias toward terminating once accepting, so runs complete
                if (eosOk && (bytes.isEmpty() || rng.nextInt(4) == 0)) {
                    completed = true;
                    break;
                }
                if (bytes.isEmpty()) break;
                int b = bytes.get(rng.nextInt(bytes.size()));
                out.write(b);
                c.advanceWith(b);
            }
            byte[] gen = out.toByteArray();
            int[] r = probe(spec, BV, gen);
            // re-feeding what we generated must never hit a rejected byte
            if (r[0] != -1) {
                check(name + " roundtrip prefix-valid run#" + run, false);
                continue;
            }
            if (completed) {
                check(name + " roundtrip accepts run#" + run, r[1] == 1);
                generated++;
            }
        }
        check(name + " roundtrip produced completed samples", generated > 0);
    }
}
