package com.qxotic.jinfer.jinja;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Behavioral tests for {@link JinjaRenderer}, the minimal Jinja2 engine used to apply GGUF
 * {@code tokenizer.chat_template} strings. No model is required — every case renders a template
 * against an in-memory context and compares the exact output string.
 *
 * <p>The bulk of the suite locks in the features real chat templates depend on (for-loops over
 * messages, object key access, filters, tests, whitespace control, macros, namespaces). A final
 * {@code knownLimitations()} group pins the current behavior of constructs this engine does NOT
 * implement like CPython Jinja2 (array indexing, list/dict literals, open-ended slices, …) so the
 * gaps are documented and a future fix trips a test instead of changing behavior silently.
 */
public final class JinjaRendererTest {

    static int failures;

    public static void main(String[] args) {
        literalsAndOutput();
        arithmetic();
        comparisonsAndLogic();
        stringConcatAndSlicing();
        filters();
        tests();
        membership();
        controlFlow();
        forLoops();
        setAndNamespace();
        macros();
        whitespaceAndComments();
        blockTrimDefaults();
        sequencesAndSlicing();
        objectAccess();
        templateFunctions();
        collectionLiteralsAndConcat();
        tojson();
        realisticChatTemplate();
        generationTag();
        compileReuse();
        unsupportedFeaturesThrow();
        lenientQuirks();
        realModelChatTemplates();

        if (failures > 0) { System.err.println("\nJinjaRendererTest: " + failures + " failures"); System.exit(1); }
        System.out.println("\nJinjaRendererTest: 0 failures");
    }

    // ── real model chat templates (resources/chat_templates/*.jinja) ──
    // Bundled, verbatim tokenizer.chat_template strings from popular models (provenance is in each
    // file's header comment). Beyond "it renders", each template is checked for structural
    // correctness against a known conversation: every turn survives and stays ordered, the system
    // message and bos_token come through, a passed tool is actually declared, the generation prompt
    // takes effect, and no value mis-renders ([object]/[function]/LitNode) leak into the output.
    // These guard the silent-corruption bugs (dropped loops, stringified ternaries, tools ignored)
    // that "does it throw?" tests miss.

    // distinctive per-turn markers (unlikely to be transformed by any template)
    static final String SYS = "SYS_q9", U1 = "USR1_q9", A1 = "AST1_q9", U2 = "USR2_q9";
    static final String BOS = "<|bos_q9|>", EOS = "<|eos_q9|>", TOOL = "get_current_weather", TOOL2 = "set_alarm";

    static void realModelChatTemplates() {
        System.out.println("-- real model chat templates --");
        List<Path> templates;
        try {
            URL dir = JinjaRendererTest.class.getResource("/chat_templates");
            if (dir == null) { check("chat_templates resources on classpath", false); return; }
            templates = new ArrayList<>();
            try (var s = Files.newDirectoryStream(Path.of(dir.toURI()), "*.jinja")) {
                for (Path p : s) templates.add(p);
            }
            templates.sort(Comparator.comparing(p -> p.getFileName().toString()));
        } catch (Exception e) {
            check("enumerate chat_templates resources (" + e + ")", false);
            return;
        }
        check("found bundled chat templates (" + templates.size() + ")", templates.size() >= 25);
        for (Path p : templates) {
            String name = p.getFileName().toString();
            String tpl;
            try { tpl = Files.readString(p); }
            catch (Exception e) { check(name + " readable", false); continue; }
            JinjaRenderer.Prog prog;
            try { prog = JinjaRenderer.parse(tpl); }
            catch (RuntimeException e) { check(name + " COMPILES (" + oneLine(e) + ")", false); continue; }
            validateTemplate(name, tpl, prog);
        }
    }

    static void validateTemplate(String name, String tpl, JinjaRenderer.Prog prog) {
        // Find a context shape this template accepts: some reject a system turn (Gemma), some take
        // content only as a multimodal parts list (MiniMax-M1). Pick the first shape that renders
        // all three conversation turns.
        String out = null;
        boolean withSystem = false, listContent = false;
        outer:
        for (boolean sys : new boolean[]{true, false}) {
            for (boolean lst : new boolean[]{false, true}) {
                String o = tryRender(prog, conv(sys, oneTool(), true, lst, false));
                if (o != null && o.contains(U1) && o.contains(A1) && o.contains(U2)) {
                    out = o; withSystem = sys; listContent = lst; break outer;
                }
            }
        }
        if (out == null) { check(name + " renders all conversation turns", false); return; }
        check(name + " renders all conversation turns", true);

        // turns stay in order
        check(name + " preserves turn order",
                out.indexOf(U1) < out.indexOf(A1) && out.indexOf(A1) < out.indexOf(U2));
        // every turn rendered EXACTLY once — guards duplicated or skipped messages (off-by-one in
        // the message loop, last-message special-casing that double-emits, etc.)
        check(name + " renders each turn exactly once",
                count(out, U1) == 1 && count(out, A1) == 1 && count(out, U2) == 1);
        // system content survives (when the template accepts a system turn)
        if (withSystem) check(name + " includes system content", out.contains(SYS));
        // no value mis-renders leaked into the text
        check(name + " no mis-rendered values",
                !out.contains("[object]") && !out.contains("[function") && !out.contains("LitNode"));
        // no UNRENDERED statement/comment tags leak. (Tool-doc strings legitimately contain `{{`
        // braces, so we only flag `{%`/`{#`, which never appear as literal output text.)
        check(name + " no unrendered tags", !out.contains("{%") && !out.contains("{#"));
        // rendering is deterministic — identical input must give identical output, guarding state
        // that leaks across renders (mutated namespaces, aliased list slices, .pop() side-effects)
        String again = tryRender(prog, conv(withSystem, oneTool(), true, listContent, false));
        check(name + " is deterministic across renders", out.equals(again));
        // bos_token is emitted as its token string (not dropped / printed as the name)
        if (tpl.contains("bos_token")) check(name + " emits bos_token value", out.contains(BOS));
        // the generation prompt actually changes the output (assistant turn opener appended)
        if (tpl.contains("add_generation_prompt")) {
            String noGen = tryRender(prog, conv(withSystem, oneTool(), false, listContent, false));
            if (noGen != null) check(name + " add_generation_prompt has effect", !noGen.equals(out));
        }
        // (Note: we deliberately don't assert a generic "thinking flag toggles output" — models
        // gate reasoning inconsistently: enable_thinking bool, a `thinking` bool, a reasoning_effort
        // enum, or `/think` markers in the content — so there's no uniform signal to flip.)
        // tools: a provided tool must be declared (assert only when tools change the output, i.e.
        // the template supports tools at all) ...
        String noTools = tryRender(prog, conv(withSystem, null, true, listContent, false));
        if (noTools != null && !noTools.equals(out)) {
            check(name + " declares the tool when tools are provided", out.contains(TOOL));
            // ... and ALL tools are declared, each once — guards a broken tool loop (only first/last)
            String two = tryRender(prog, conv(withSystem, twoTools(), true, listContent, false));
            if (two != null) check(name + " declares every tool exactly once",
                    count(two, TOOL) == 1 && count(two, TOOL2) == 1);
        }

        validateEdgeCases(name, prog, listContent);
    }

    /** Minimal / awkward inputs that commonly break loops and last-message handling. */
    static void validateEdgeCases(String name, JinjaRenderer.Prog prog, boolean listContent) {
        // a single user message (no prior turns) is the most basic valid input — every chat
        // template must handle it without dropping the message
        var single = map("messages", list(map("role", "user", "content", body("Solo " + U1, listContent))),
                "add_generation_prompt", true, "bos_token", BOS, "eos_token", EOS, "tools", null, "enable_thinking", false);
        String solo = tryRender(prog, single);
        check(name + " single-message conversation renders the message", solo != null && solo.contains(U1));

        // content with quotes, braces, backslash, newline and unicode must not crash the renderer
        // or silently swallow the message (guards over-eager stripping / escaping bugs)
        String tricky = "q\"'{}<>\\b\n\tüñ€ " + U2;
        var special = map("messages", list(
                map("role", "user", "content", body("hi", listContent)),
                map("role", "assistant", "content", body("ok", listContent)),
                map("role", "user", "content", body(tricky, listContent))),
            "add_generation_prompt", true, "bos_token", BOS, "eos_token", EOS, "tools", null, "enable_thinking", false);
        String sp = tryRender(prog, special);
        check(name + " handles special characters in content", sp != null && sp.contains(U2));
    }

    /** A system?/user/assistant/user conversation. {@code tools} null = none; {@code listContent}
     *  wraps each content as a multimodal parts list. */
    static Map<String,Object> conv(boolean withSystem, List<Object> tools, boolean genPrompt, boolean listContent, boolean thinking) {
        var msgs = new ArrayList<Object>();
        if (withSystem) msgs.add(map("role", "system", "content", body("You are helpful. " + SYS, listContent)));
        msgs.add(map("role", "user", "content", body("Question one " + U1, listContent)));
        msgs.add(map("role", "assistant", "content", body("Answer one " + A1, listContent)));
        msgs.add(map("role", "user", "content", body("Question two " + U2, listContent)));
        var ctx = map("messages", msgs, "add_generation_prompt", genPrompt,
                "bos_token", BOS, "eos_token", EOS, "enable_thinking", thinking);
        ctx.put("tools", tools);
        return ctx;
    }

    static Object body(String text, boolean listContent) {
        return listContent ? list(map("type", "text", "text", text)) : text;
    }

    static List<Object> oneTool() { return list(tool(TOOL, "Get the current weather", "location")); }
    static List<Object> twoTools() { return list(tool(TOOL, "Get the current weather", "location"),
                                                 tool(TOOL2, "Set an alarm", "time")); }

    static Object tool(String fn, String desc, String param) {
        return map("type", "function", "function", map(
                "name", fn, "description", desc,
                "parameters", map("type", "object",
                        "properties", map(param, map("type", "string", "description", param + " value")),
                        "required", list(param))));
    }

    /** Count non-overlapping occurrences of {@code sub} in {@code s}. */
    static int count(String s, String sub) {
        int n = 0, i = 0;
        while ((i = s.indexOf(sub, i)) >= 0) { n++; i += sub.length(); }
        return n;
    }

    static String tryRender(JinjaRenderer.Prog prog, Map<String,Object> ctx) {
        try { return JinjaRenderer.render(prog, ctx); }
        catch (RuntimeException e) { return null; }
    }

    // ── harness ──────────────────────────────────────────────────

    /** Render {@code tpl} against {@code ctx} and assert the output equals {@code expected}. */
    static void eq(String tpl, Map<String,Object> ctx, String expected) {
        String got;
        try {
            got = JinjaRenderer.render(tpl, ctx);
        } catch (Throwable t) {
            failures++;
            System.err.println("FAIL: " + show(tpl) + " threw " + t);
            return;
        }
        if (expected.equals(got)) {
            System.out.println("ok: " + show(tpl) + " => " + show(got));
        } else {
            failures++;
            System.err.println("FAIL: " + show(tpl) + "\n  expected [" + show(expected) + "]\n  got      [" + show(got) + "]");
        }
    }

    static void eq(String tpl, String expected) { eq(tpl, Map.of(), expected); }

    static void check(String what, boolean ok) {
        if (ok) System.out.println("ok: " + what);
        else { failures++; System.err.println("FAIL: " + what); }
    }

    /** Assert that rendering {@code tpl} raises (parse or eval error). */
    static void throwsErr(String what, String tpl) { throwsErr(what, tpl, Map.of()); }

    static void throwsErr(String what, String tpl, Map<String,Object> ctx) {
        try {
            JinjaRenderer.render(tpl, ctx);
            failures++;
            System.err.println("FAIL: " + what + " (expected an exception, none thrown)");
        } catch (RuntimeException e) {
            System.out.println("ok: " + what + " (threw: " + e.getMessage() + ")");
        }
    }

    static String render(String tpl, Map<String,Object> ctx) { return JinjaRenderer.render(tpl, ctx); }
    static String render(String tpl) { return JinjaRenderer.render(tpl, Map.of()); }

    static String show(String s) { return s.replace("\n", "\\n").replace("\t", "\\t"); }

    static String oneLine(Throwable t) { String m = t.getMessage(); return show(m == null ? t.toString() : m); }

    /** Ordered map literal: {@code map("a", 1, "b", 2)}. */
    static Map<String,Object> map(Object... kv) {
        var m = new LinkedHashMap<String,Object>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    static List<Object> list(Object... items) {
        return new ArrayList<>(List.of(items));
    }

    // ── {% generation %} tag (transformers assistant-span extension; SmolLM3 uses it) ──

    static void generationTag() {
        System.out.println("-- {% generation %} tag --");
        // transparent: the body renders exactly as if the tags weren't there
        eq("{% generation %}hello{% endgeneration %}", "hello");
        eq("A{% generation %}B{{ x }}{% endgeneration %}C", Map.of("x", "!"), "AB!C");
        // nested inside for / if
        eq("{% for m in items %}{% generation %}{{ m }}{% endgeneration %}{% endfor %}",
                Map.of("items", List.of("a", "b")), "ab");
        eq("{% if flag %}{% generation %}X{% endgeneration %}{% endif %}", Map.of("flag", true), "X");
        // unclosed -> a clear parse error, like an unterminated for/if
        throwsErr("unclosed {% generation %}", "{% generation %}oops");
        // spans: (start,end) char offsets bound exactly the generated substring
        var r = JinjaRenderer.renderWithSpans("A{% generation %}BC{% endgeneration %}D", Map.of());
        check("generation span: text == ABCD", r.text().equals("ABCD"));
        check("generation span: exactly one span", r.generationSpans().size() == 1);
        int[] sp = r.generationSpans().size() == 1 ? r.generationSpans().get(0) : new int[]{-1, -1};
        check("generation span: [1,3] bounds 'BC'",
                sp[0] == 1 && sp[1] == 3 && r.text().substring(Math.max(0, sp[0]), Math.max(0, sp[1])).equals("BC"));
    }

    // ── literals & output ────────────────────────────────────────

    static void literalsAndOutput() {
        System.out.println("-- literals & output --");
        eq("plain text, no tags", "plain text, no tags");
        eq("{{ 42 }}", "42");
        eq("{{ 3.14 }}", "3.14");
        eq("{{ 2.0 }}", "2");                       // whole floats render without a trailing .0
        eq("{{ -5 }}", "-5");
        eq("{{ 'single' }}", "single");
        eq("{{ \"double\" }}", "double");
        eq("{{ True }} {{ False }} {{ None }}", "True False None");
        eq("{{ missing }}", "");                    // undefined variable renders empty
        eq("{{ name }}", map("name", "World"), "World");
        eq("a={{ 1 }}, b={{ 2 }}", "a=1, b=2");
        eq("Hello, {{ name }}!", map("name", "Ada"), "Hello, Ada!");
    }

    // ── arithmetic ───────────────────────────────────────────────

    static void arithmetic() {
        System.out.println("-- arithmetic --");
        eq("{{ 2 + 3 }}", "5");
        eq("{{ 10 - 4 }}", "6");
        eq("{{ 6 * 7 }}", "42");
        eq("{{ 3 / 2 }}", "1.5");
        eq("{{ 9 / 3 }}", "3");                     // exact division renders as integer
        eq("{{ 10 % 3 }}", "1");
        eq("{{ 2 + 3 * 4 }}", "14");               // precedence: * before +
        eq("{{ (2 + 3) * 4 }}", "20");             // parentheses override
        eq("{{ 10 - 2 - 3 }}", "5");               // left-associative
        eq("{{ n + 1 }}", map("n", 7), "8");
        eq("{{ -n }}", map("n", 7), "-7");
        eq("{{ +n }}", map("n", 7), "7");
    }

    // ── comparisons & boolean logic ──────────────────────────────

    static void comparisonsAndLogic() {
        System.out.println("-- comparisons & logic --");
        eq("{{ 1 == 1 }}", "True");
        eq("{{ 1 == 2 }}", "False");
        eq("{{ 1 != 2 }}", "True");
        eq("{{ 2 > 1 }} {{ 1 < 2 }} {{ 2 >= 2 }} {{ 1 <= 0 }}", "True True True False");
        eq("{{ 'a' == 'a' }} {{ 'a' != 'b' }}", "True True");
        eq("{{ True and False }} {{ True or False }} {{ not False }}", "False True True");
        eq("{{ n > 5 and n < 10 }}", map("n", 7), "True");
        eq("{{ n > 5 and n < 10 }}", map("n", 42), "False");
        // and/or return the OPERAND value (Python/Jinja), not a coerced bool
        eq("{{ '' or 'fallback' }}", "fallback");
        eq("{{ 'keep' or 'other' }}", "keep");
        eq("{{ missing or 'dflt' }}", "dflt");
        eq("{{ name or 'dflt' }}", map("name", "Ada"), "Ada");
        eq("{{ 'a' and 'b' }}", "b");
        eq("{{ '' and 'b' }}", "");
        // the canonical default idiom: obj.get(k) or fallback
        eq("{{ user.get('nick') or user['name'] }}", map("user", map("name", "Bob")), "Bob");
        // both capitalized and lowercase literals are accepted
        eq("{{ true }} {{ false }} {{ none }}", "True False None");
        eq("{{ true == True }} {{ none is none }}", "True True");
    }

    // ── string concat & slicing ──────────────────────────────────

    static void stringConcatAndSlicing() {
        System.out.println("-- string concat & slicing --");
        eq("{{ 'a' ~ 'b' }}", "ab");                // ~ concatenation
        eq("{{ 'x' ~ 1 }}", "x1");                  // ~ stringifies operands
        eq("{{ 'a' + 'b' }}", "ab");                // + concatenates when either side is a string
        eq("{{ greeting ~ '!' }}", map("greeting", "hi"), "hi!");
        eq("{{ 'hello'[1:4] }}", "ell");           // substring slice
        eq("{{ 'hello'[:3] }}", "hel");            // open start
        eq("{{ 'hello'[0:2] }}", "he");
        eq("{{ 'hello'[2:] }}", "llo");            // open-ended stop
        eq("{{ 'hello'[-3:] }}", "llo");           // negative start
        eq("{{ 'hello'[:-2] }}", "hel");           // negative stop
        eq("{{ 'hello'[::-1] }}", "olleh");        // reverse via negative step
        eq("{{ 'hi'[0] }} {{ 'hi'[1] }} {{ 'hi'[-1] }}", "h i i"); // character indexing
        // string methods (standard parenthesized form, with arguments)
        eq("{{ 'HeLLo'.lower() }}", "hello");
        eq("{{ 'HeLLo'.upper() }}", "HELLO");
        eq("{{ '  trim me  '.strip() }}", "trim me");
        eq("{{ 'a,b,c'.split(',') | join('|') }}", "a|b|c");
        eq("{{ 'hello world'.startswith('hello') }}", "True");
        eq("{{ 'hello world'.endswith('world') }}", "True");
        eq("{{ 'foobar'.replace('o', '0') }}", "f00bar");
    }

    // ── filters ──────────────────────────────────────────────────

    static void filters() {
        System.out.println("-- filters --");
        eq("{{ 'WIDE' | lower }}", "wide");
        eq("{{ 'wide' | upper }}", "WIDE");
        eq("{{ '  pad  ' | trim }}", "pad");
        eq("{{ name | length }}", map("name", "abcd"), "4");
        eq("{{ xs | length }}", map("xs", list(1, 2, 3)), "3");
        eq("{{ obj | length }}", map("obj", map("a", 1, "b", 2)), "2");
        eq("{{ xs | join(', ') }}", map("xs", list("a", "b", "c")), "a, b, c");
        eq("{{ xs | join }}", map("xs", list("a", "b", "c")), "abc");
        eq("{{ 'a,b,c' | split(',') | join('|') }}", "a|b|c");
        eq("{{ name | startswith('Wo') }}", map("name", "World"), "True");
        eq("{{ name | endswith('ld') }}", map("name", "World"), "True");
        eq("{{ 'text' | string }}", "text");
        // default: substitutes only when the value is undefined
        eq("{{ missing | default('fallback') }}", "fallback");
        eq("{{ name | default('fallback') }}", map("name", "present"), "present");
        eq("{{ missing | default }}", "");
        // chained filters
        eq("{{ '  Mixed  ' | trim | upper }}", "MIXED");
    }

    // ── tests (is ...) ───────────────────────────────────────────

    static void tests() {
        System.out.println("-- tests --");
        eq("{{ name is defined }}", map("name", "x"), "True");
        eq("{{ missing is defined }}", "False");
        eq("{{ missing is undefined }}", "True");
        eq("{{ nul is none }}", map("nul", (Object) null), "True");
        eq("{{ name is none }}", map("name", "x"), "False");
        eq("{{ s is string }} {{ n is number }} {{ b is boolean }}",
                map("s", "x", "n", 1, "b", true), "True True True");
        eq("{{ n is integer }} {{ f is float }}", map("n", 3, "f", 1.5), "True True");
        eq("{{ d is mapping }} {{ a is sequence }}",
                map("d", map("k", 1), "a", list(1)), "True True");
        eq("{{ 4 is even }} {{ 3 is odd }} {{ 3 is even }}", "True True False");
        eq("{{ x is not none }}", map("x", 1), "True");
        eq("{{ n is not number }}", map("n", 1), "False");
    }

    // ── membership (in / not in) ─────────────────────────────────

    static void membership() {
        System.out.println("-- membership --");
        eq("{{ 'b' in xs }}", map("xs", list("a", "b", "c")), "True");
        eq("{{ 'z' in xs }}", map("xs", list("a", "b", "c")), "False");
        eq("{{ 'z' not in xs }}", map("xs", list("a", "b", "c")), "True");
        eq("{{ 'ell' in word }}", map("word", "hello"), "True");          // substring membership
        eq("{{ 'role' in obj }}", map("obj", map("role", "admin")), "True"); // object key membership
        eq("{{ 'nope' in obj }}", map("obj", map("role", "admin")), "False");
    }

    // ── control flow ─────────────────────────────────────────────

    static void controlFlow() {
        System.out.println("-- control flow --");
        eq("{% if flag %}yes{% endif %}", map("flag", true), "yes");
        eq("{% if flag %}yes{% endif %}", map("flag", false), "");
        eq("{% if flag %}yes{% else %}no{% endif %}", map("flag", false), "no");
        // elif chains, with and without a trailing else, pick exactly one branch
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% endif %}", map("n", 99), "big");
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% endif %}", map("n", 7), "mid");
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% endif %}", map("n", 1), "");
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% else %}small{% endif %}", map("n", 99), "big");
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% else %}small{% endif %}", map("n", 7), "mid");
        eq("{% if n > 10 %}big{% elif n > 5 %}mid{% else %}small{% endif %}", map("n", 1), "small");
        // multi-elif cascade with else
        eq("{% if g == 'A' %}1{% elif g == 'B' %}2{% elif g == 'C' %}3{% else %}?{% endif %}", map("g", "C"), "3");
        eq("{% if g == 'A' %}1{% elif g == 'B' %}2{% elif g == 'C' %}3{% else %}?{% endif %}", map("g", "Z"), "?");
        // ternary expression
        eq("{{ 'on' if flag else 'off' }}", map("flag", true), "on");
        eq("{{ 'on' if flag else 'off' }}", map("flag", false), "off");
        eq("{{ 'truthy' if name else 'empty' }}", map("name", ""), "empty");
        // ternary yields the operand VALUE, not its rendered text — must work for lists/objects
        eq("{% set xs = a if flag else b %}{{ xs | join('-') }}", map("flag", true, "a", list(1, 2), "b", list(9)), "1-2");
        eq("{% set xs = a if flag else b %}{{ xs | join('-') }}", map("flag", false, "a", list(1, 2), "b", list(9)), "9");
        eq("{% set m = messages[1:] if drop else messages %}{{ m | length }}",
                map("drop", true, "messages", list("s", "u", "a")), "2");
        // a ternary with no else yields undefined when the test is false
        eq("{{ ('x' if flag) | default('none') }}", map("flag", false), "none");
    }

    // ── for loops ────────────────────────────────────────────────

    static void forLoops() {
        System.out.println("-- for loops --");
        eq("{% for x in xs %}{{ x }};{% endfor %}", map("xs", list("a", "b", "c")), "a;b;c;");
        eq("[{% for x in xs %}{{ x }}{% endfor %}]", map("xs", list()), "[]"); // empty iterable
        // loop variables
        eq("{% for x in xs %}{{ loop.index }}{% endfor %}", map("xs", list("a", "b", "c")), "123");
        eq("{% for x in xs %}{{ loop.index0 }}{% endfor %}", map("xs", list("a", "b", "c")), "012");
        eq("{% for x in xs %}{{ loop.length }}{% endfor %}", map("xs", list("a", "b")), "22");
        eq("{% for x in xs %}{% if loop.first %}<{% endif %}{{ x }}{% if loop.last %}>{% endif %}{% endfor %}",
                map("xs", list("a", "b", "c")), "<abc>");
        eq("{% for x in xs %}{{ x }}{% if not loop.last %}, {% endif %}{% endfor %}",
                map("xs", list("a", "b", "c")), "a, b, c");
        // iterating an object's items()
        eq("{% for k, v in obj.items() %}{{ k }}={{ v }};{% endfor %}",
                map("obj", map("a", 1, "b", 2)), "a=1;b=2;");
        // nested loops
        eq("{% for r in rows %}{% for c in r %}{{ c }}{% endfor %}|{% endfor %}",
                map("rows", list(list("1", "2"), list("3", "4"))), "12|34|");
    }

    // ── set & namespace ──────────────────────────────────────────

    static void setAndNamespace() {
        System.out.println("-- set & namespace --");
        eq("{% set x = 5 %}{{ x }}", "5");
        eq("{% set greeting = 'Hi ' ~ name %}{{ greeting }}", map("name", "Sam"), "Hi Sam");
        eq("{% set total = a + b %}{{ total }}", map("a", 3, "b", 4), "7");
        // namespace() with a mutable field updated across statements (canonical Jinja pattern)
        eq("{% set ns = namespace(found=False) %}{% set ns.found = True %}{{ ns.found }}", "True");
        eq("{% set ns = namespace(role='guest') %}{% set ns.role = 'admin' %}{{ ns.role }}", "admin");
        // namespace fields keep their value type: a numeric counter ADDS, not concatenates
        eq("{% set ns = namespace(count=0) %}{% for x in xs %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}",
                map("xs", list("a", "b", "c")), "3");
        // boolean field flips correctly and reads back as a real boolean
        eq("{% set ns = namespace(seen=False) %}{% for x in xs %}{% if x == 'b' %}{% set ns.seen = True %}{% endif %}{% endfor %}{{ ns.seen }}",
                map("xs", list("a", "b", "c")), "True");
    }

    // ── macros ───────────────────────────────────────────────────

    static void macros() {
        System.out.println("-- macros --");
        eq("{% macro greet(n) %}Hi {{ n }}!{% endmacro %}{{ greet('Bob') }}", "Hi Bob!");
        eq("{% macro tag(name, val) %}<{{ name }}>{{ val }}</{{ name }}>{% endmacro %}{{ tag('b', 'x') }}",
                "<b>x</b>");
        eq("{% macro line(x) %}- {{ x }}\n{% endmacro %}{% for i in xs %}{{ line(i) }}{% endfor %}",
                map("xs", list("one", "two")), "- one\n- two\n");
    }

    // ── whitespace control & comments ────────────────────────────

    static void whitespaceAndComments() {
        System.out.println("-- whitespace & comments --");
        eq("a {# this is a comment #}b", "a b");
        eq("x{# c #}y", "xy");
        // statement-level trim markers ({%- ... -%}) collapse whitespace between the tags
        eq("{%- if true_flag -%} Y {%- endif -%}", map("true_flag", true), "Y");
        eq("{%- for x in xs -%} {{ x }} {%- endfor -%}", map("xs", list("a", "b")), "ab");
    }

    // ── sequence indexing, slicing & filters ─────────────────────

    static void sequencesAndSlicing() {
        System.out.println("-- sequences: indexing, slicing, filters --");
        var ctx = map("xs", list("a", "b", "c", "d"));
        // integer indexing (Python-style negatives)
        eq("{{ xs[0] }} {{ xs[2] }} {{ xs[-1] }} {{ xs[-2] }}", ctx, "a c d c");
        eq("{{ xs[99] }}", ctx, "");               // out of range -> undefined -> empty
        // indexing the result of an object lookup
        eq("{{ msgs[0]['role'] }}/{{ msgs[-1]['content'] }}",
                map("msgs", list(map("role", "system", "content", "x"), map("role", "user", "content", "hi"))),
                "system/hi");
        // array slicing
        eq("{{ xs[1:3] | join(',') }}", ctx, "b,c");
        eq("{{ xs[1:] | join(',') }}", ctx, "b,c,d");
        eq("{{ xs[:2] | join(',') }}", ctx, "a,b");
        eq("{{ xs[:-1] | join(',') }}", ctx, "a,b,c");
        eq("{{ xs[::-1] | join(',') }}", ctx, "d,c,b,a"); // reverse
        eq("{{ xs[::2] | join(',') }}", ctx, "a,c");      // step
        // first / last / length / list filters
        eq("{{ xs | first }}/{{ xs | last }}/{{ xs | length }}", ctx, "a/d/4");
        // selectattr / rejectattr over a list of objects
        var people = map("people", list(
                map("name", "Ada", "role", "admin"),
                map("name", "Bob", "role", "user"),
                map("name", "Cy", "role", "admin")));
        eq("{{ people | selectattr('role', 'equalto', 'admin') | list | length }}", people, "2");
        eq("{% for p in people | selectattr('role', 'equalto', 'admin') %}{{ p.name }};{% endfor %}", people, "Ada;Cy;");
        eq("{% for p in people | rejectattr('role', 'equalto', 'admin') %}{{ p.name }};{% endfor %}", people, "Bob;");
    }

    // ── object / dict access & methods ───────────────────────────

    // ── trim_blocks / lstrip_blocks (HF apply_chat_template defaults) ───────

    static void blockTrimDefaults() {
        System.out.println("-- trim_blocks / lstrip_blocks defaults --");
        // trim_blocks: a bare %} swallows exactly one following newline
        eq("{% if true %}\nA{% endif %}\n", map(), "A");
        eq("{% if true %}\r\nA{% endif %}", map(), "A");                    // and \r\n as one
        eq("{% if true %}\n\nA{% endif %}", map(), "\nA");                 // only ONE newline
        // comments trim the same way
        eq("{# note #}\nA", map(), "A");
        // lstrip_blocks: whitespace from line start up to a block tag is stripped
        eq("A\n    {% if true %}B{% endif %}", map(), "A\nB");
        eq("A\n\t{% if true %}B{% endif %}", map(), "A\nB");
        // ...but NOT when non-whitespace precedes the tag on the line
        eq("abc {% if true %}B{% endif %}", map(), "abc B");
        // ...and NOT for output tags
        eq("A\n    {{ 'B' }}", map(), "A\n    B");
        // flags do NOT leak across an intervening tag (reference jinja2: trim/strip applies only
        // to text DIRECTLY after the tag that set it)
        eq("{% if true %}{{ 'A' }}\nB{% endif %}", map(), "A\nB");          // newline after }} kept
        eq("{% if true -%}{{ 'A' }} B{% endif %}", map(), "A B");            // -%} does not strip past {{ }}
        eq("{# c #}{{ 'A' }}\nB", map(), "A\nB");                            // comment trim dies at {{
        // lstrip only strips whitespace that starts at a LINE start
        eq("{{ 'A' }} {% if true %}B{% endif %}", map(), "A B");             // mid-line space after }} kept
        eq("{{ 'A' }}\n  {% if true %}B{% endif %}", map(), "A\nB");         // but a real line start strips
        eq("  {% if true %}B{% endif %}", map(), "B");                       // template-start whitespace strips
        // explicit markers still win over the defaults
        eq("{% if true -%}\n   A{% endif %}", map(), "A");                   // -%} strips ALL leading ws
        eq("A   {%- if true %}B{% endif %}", map(), "AB");                    // {%- strips preceding ws
        // keep_trailing_newline=false: exactly one trailing newline is dropped - at the
        // template() entry point (what GgufTokenizer renders through), not the raw render()
        eqTemplate("A\n", "A");
        eqTemplate("A\n\n", "A\n");
    }

    /** eq via the public template() entry point (keep_trailing_newline applies there). */
    static void eqTemplate(String tpl, String expected) {
        String got = JinjaRenderer.template(tpl).render(map());
        if (expected.equals(got)) {
            System.out.println("ok: template(" + show(tpl) + ") => " + show(got));
        } else {
            failures++;
            System.err.println("FAIL: template(" + show(tpl) + ")\n  expected [" + show(expected) + "]\n  got      [" + show(got) + "]");
        }
    }

    static void objectAccess() {
        System.out.println("-- object access --");
        var user = map("role", "admin", "age", 30, "address", map("city", "NYC"));
        eq("{{ user.role }}", map("user", user), "admin");
        eq("{{ user['role'] }}", map("user", user), "admin");      // computed string key
        eq("{{ user.age }}", map("user", user), "30");
        eq("{{ user.address.city }}", map("user", user), "NYC");   // nested dot
        eq("{{ user['address']['city'] }}", map("user", user), "NYC"); // nested computed
        eq("{{ user.missing }}", map("user", user), "");           // absent key -> empty
        // dict methods (parenthesized — they resolve to builtin functions)
        eq("{{ obj.keys() | join(',') }}", map("obj", map("a", 1, "b", 2)), "a,b");
        eq("{{ obj.values() | join(',') }}", map("obj", map("a", 1, "b", 2)), "1,2");
        eq("{{ obj.get('a') }}", map("obj", map("a", 1)), "1");
        eq("{{ obj.get('zzz', 'def') }}", map("obj", map("a", 1)), "def");
    }

    // ── template functions (raise_exception, strftime_now) ───────

    static void templateFunctions() {
        System.out.println("-- template functions --");
        // raise_exception(msg) aborts rendering with the given message (chat templates use it to
        // reject malformed conversations)
        try {
            JinjaRenderer.render("{% if true_flag %}{{ raise_exception('bad input') }}{% endif %}", map("true_flag", true));
            failures++; System.err.println("FAIL: raise_exception should throw");
        } catch (RuntimeException e) {
            check("raise_exception throws with its message",
                    e.getMessage() != null && e.getMessage().contains("bad input"));
        }
        // strftime_now(fmt) renders the current time; just assert the format is applied (4-digit year)
        String year = JinjaRenderer.render("{{ strftime_now('%Y') }}", Map.of());
        check("strftime_now('%Y') yields a 4-digit year (got " + year + ")", year.matches("\\d{4}"));
        eq("{{ strftime_now('literal %% sign') }}", "literal % sign");
        // range([start,] stop[, step]) -> list of ints, stop exclusive (Python/Jinja semantics)
        eq("{% for i in range(3) %}{{ i }},{% endfor %}", "0,1,2,");
        eq("{% for i in range(2, 5) %}{{ i }},{% endfor %}", "2,3,4,");
        eq("{% for i in range(0, 10, 2) %}{{ i }},{% endfor %}", "0,2,4,6,8,");
        eq("{% for i in range(messages|length) %}{{ messages[i] }};{% endfor %}",
                map("messages", List.of("a", "b")), "a;b;");
    }

    // ── tojson ───────────────────────────────────────────────────

    static void tojson() {
        System.out.println("-- tojson --");
        eq("{{ s | tojson }}", map("s", "hi"), "\"hi\"");
        eq("{{ n | tojson }}", map("n", 42), "42");
        eq("{{ f | tojson }}", map("f", 1.5), "1.5");
        eq("{{ b | tojson }}", map("b", true), "true");          // JSON casing, not Python True
        eq("{{ nul | tojson }}", map("nul", (Object) null), "null");
        eq("{{ xs | tojson }}", map("xs", list(1, 2, 3)), "[1, 2, 3]");
        eq("{{ obj | tojson }}", map("obj", map("role", "admin", "age", 30)),
                "{\"role\": \"admin\", \"age\": 30}");
        // string values are JSON-escaped
        eq("{{ s | tojson }}", map("s", "a\"b"), "\"a\\\"b\"");
    }

    // ── a realistic ChatML-style template ────────────────────────

    static void realisticChatTemplate() {
        System.out.println("-- realistic chat template --");
        String tpl = "{% for m in messages %}"
                + "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                + "{% endfor %}"
                + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        var messages = list(
                map("role", "system", "content", "Be nice"),
                map("role", "user", "content", "Hello"));
        eq(tpl, map("messages", messages, "add_generation_prompt", true),
                "<|im_start|>system\nBe nice<|im_end|>\n"
                        + "<|im_start|>user\nHello<|im_end|>\n"
                        + "<|im_start|>assistant\n");
        // without the generation prompt
        eq(tpl, map("messages", messages, "add_generation_prompt", false),
                "<|im_start|>system\nBe nice<|im_end|>\n"
                        + "<|im_start|>user\nHello<|im_end|>\n");
        // a bos_token prefix + role dispatch via if/elif, as several real templates do
        String tpl2 = "{{ bos_token }}{% for m in messages %}"
                + "{% if m['role'] == 'user' %}[USER] {{ m['content'] }}\n"
                + "{% elif m['role'] == 'assistant' %}[BOT] {{ m['content'] }}\n"
                + "{% endif %}{% endfor %}";
        eq(tpl2, map("bos_token", "<s>", "messages",
                        list(map("role", "user", "content", "hi"),
                             map("role", "assistant", "content", "yo"))),
                "<s>[USER] hi\n[BOT] yo\n");
    }

    // ── compile once, render many ────────────────────────────────

    static void compileReuse() {
        System.out.println("-- compile reuse --");
        JinjaRenderer.Prog prog = JinjaRenderer.parse("Hi {{ name }} ({{ n }})");
        check("compiled program renders #1",
                "Hi Ada (1)".equals(JinjaRenderer.render(prog, map("name", "Ada", "n", 1))));
        check("compiled program renders #2 with new context",
                "Hi Bob (2)".equals(JinjaRenderer.render(prog, map("name", "Bob", "n", 2))));
    }

    // ── unsupported features fail loud ───────────────────────────
    // Constructs this minimal engine does not implement raise an exception (rather than silently
    // mis-rendering the prompt). These tests pin that fail-loud contract.

    static void collectionLiteralsAndConcat() {
        System.out.println("-- collection literals & concatenation --");
        // list literals (evaluated, incl. variable elements)
        eq("{{ [1, 2, 3] | join('-') }}", "1-2-3");
        eq("{{ [a, b, 'z'] | join(',') }}", map("a", "x", "b", "y"), "x,y,z");
        eq("{% for x in [10, 20, 30] %}{{ x }};{% endfor %}", "10;20;30;");
        eq("{{ [] | length }}", "0");
        // tuple literals behave like lists
        eq("{{ (1, 2, 3) | join('-') }}", "1-2-3");
        // dict literals (string and identifier keys)
        eq("{{ {'a': 1, 'b': 2} | tojson }}", "{\"a\": 1, \"b\": 2}");
        eq("{% set d = {'x': 5, 'y': 6} %}{{ d['x'] }}+{{ d['y'] }}", "5+6");
        // chained ~ concatenation
        eq("{{ 'a' ~ 'b' ~ 'c' }}", "abc");
        eq("{{ a ~ '-' ~ b ~ '-' ~ 42 }}", map("a", "x", "b", "y"), "x-y-42");
        // adjacent string-literal concatenation (Python/Jinja2 style)
        eq("{{ 'foo' 'bar' 'baz' }}", "foobar" + "baz");
        // str() of a list/dict is the Python repr (strings quoted, True/False/None capitalized) —
        // models trained on `tools | string` depend on this exact shape
        eq("{{ [1, 'x', true] }}", "[1, 'x', True]");
        eq("{{ {'a': 1, 'b': 'x'} }}", "{'a': 1, 'b': 'x'}");
        eq("{{ {'k': 'v'} | string }}", "{'k': 'v'}");
        eq("{{ {'a': [1, 2], 'n': none} }}", "{'a': [1, 2], 'n': None}"); // nested
        eq("{{ obj | string }}", map("obj", map("name", "get_weather")), "{'name': 'get_weather'}");
    }

    static void unsupportedFeaturesThrow() {
        System.out.println("-- unsupported features throw --");
        var xs = map("xs", list("a", "b", "c"));
        // unknown filter / function names
        throwsErr("unknown filter throws", "{{ 'x' | no_such_filter }}");
        throwsErr("unknown function throws", "{{ no_such_function() }}");
        // loop control statements
        throwsErr("{% break %} throws", "{% for x in xs %}{{ x }}{% break %}{% endfor %}", xs);
        throwsErr("{% continue %} throws", "{% for x in xs %}{{ x }}{% continue %}{% endfor %}", xs);
        // integer division // is not supported
        throwsErr("integer division `//` throws", "{{ 7 // 2 }}");
    }

    // ── remaining lenient quirks (render, do NOT throw) ──────────
    // These differ from CPython Jinja2 but render leniently rather than producing obviously-broken
    // output, so they stay best-effort instead of raising. Pinned to document the behavior.

    static void lenientQuirks() {
        System.out.println("-- lenient quirks (no throw) --");
        // expression-level trim markers ({{- ... -}}) do NOT strip surrounding whitespace
        // (only statement-level {%- ... -%} markers do)
        // whitespace-control markers, reference jinja2 semantics: {{- strips the
        // PRECEDING whitespace, -}} the FOLLOWING (was swapped before the fix).
        eq("a {{- 'b' }}", "ab");
        eq("{{ 'a' -}} b", "ab");
        eq("a\n{%- if true %}b{% endif %}", "ab");
        eq("a\n\n{#- c #}\n{%- if true %}b{% endif %}", "ab");
    }
}
