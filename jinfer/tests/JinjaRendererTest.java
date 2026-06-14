package com.llama4j;

import java.util.ArrayList;
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
        sequencesAndSlicing();
        objectAccess();
        templateFunctions();
        tojson();
        realisticChatTemplate();
        compileReuse();
        unsupportedFeaturesThrow();
        lenientQuirks();

        if (failures > 0) { System.err.println("\nJinjaRendererTest: " + failures + " failures"); System.exit(1); }
        System.out.println("\nJinjaRendererTest: 0 failures");
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

    /** Ordered map literal: {@code map("a", 1, "b", 2)}. */
    static Map<String,Object> map(Object... kv) {
        var m = new LinkedHashMap<String,Object>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    static List<Object> list(Object... items) {
        return new ArrayList<>(List.of(items));
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
        JinjaRenderer.Prog prog = JinjaRenderer.compile("Hi {{ name }} ({{ n }})");
        check("compiled program renders #1",
                "Hi Ada (1)".equals(JinjaRenderer.render(prog, map("name", "Ada", "n", 1))));
        check("compiled program renders #2 with new context",
                "Hi Bob (2)".equals(JinjaRenderer.render(prog, map("name", "Bob", "n", 2))));
    }

    // ── unsupported features fail loud ───────────────────────────
    // Constructs this minimal engine does not implement raise an exception (rather than silently
    // mis-rendering the prompt). These tests pin that fail-loud contract.

    static void unsupportedFeaturesThrow() {
        System.out.println("-- unsupported features throw --");
        var xs = map("xs", list("a", "b", "c"));
        // list/dict literals with elements
        throwsErr("list literal throws", "{{ [1, 2, 3] | join('-') }}");
        throwsErr("dict literal throws", "{{ {'a': 1} | tojson }}");
        // calling a value that isn't a function
        throwsErr("calling a non-function throws", "{{ n() }}", map("n", 5));
        // unknown filter / function names
        throwsErr("unknown filter throws", "{{ 'x' | no_such_filter }}");
        throwsErr("unknown function throws", "{{ no_such_function() }}");
        // loop control statements
        throwsErr("{% break %} throws", "{% for x in xs %}{{ x }}{% break %}{% endfor %}", xs);
        throwsErr("{% continue %} throws", "{% for x in xs %}{{ x }}{% continue %}{% endfor %}", xs);
        // unparseable operators
        throwsErr("chained `~` throws", "{{ 'a' ~ 'b' ~ 'c' }}");
        throwsErr("integer division `//` throws", "{{ 7 // 2 }}");
    }

    // ── remaining lenient quirks (render, do NOT throw) ──────────
    // These differ from CPython Jinja2 but render leniently rather than producing obviously-broken
    // output, so they stay best-effort instead of raising. Pinned to document the behavior.

    static void lenientQuirks() {
        System.out.println("-- lenient quirks (no throw) --");
        // and/or yield a boolean, not the operand value (CPython returns the operand)
        eq("{{ '' or 'x' }}", "True");
        eq("{{ 'a' and 'b' }}", "True");
        // lowercase python keywords aren't literals; only True/False/None are (these read as
        // undefined and render empty)
        eq("{{ true }}", "");
        // expression-level trim markers ({{- ... -}}) do NOT strip surrounding whitespace
        // (only statement-level {%- ... -%} markers do)
        eq("a {{- 'b' }}", "a b");
        eq("{{ 'a' -}} b", "a b");
    }
}
