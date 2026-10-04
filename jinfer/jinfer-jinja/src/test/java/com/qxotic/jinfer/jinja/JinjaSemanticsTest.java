package com.qxotic.jinfer.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Jinja2 semantics that chat templates lean on, each pinned to what Jinja2 itself renders. Every
 * case here used to render differently (a dropped kwarg, a list that became a number, a loop
 * without its else, a float printed as an int) and changed the prompt bytes without an error.
 */
class JinjaSemanticsTest {

    private static String render(String template) {
        return JinjaRenderer.renderWithSpans(template, new HashMap<>()).text();
    }

    private static String render(String template, Map<String, Object> context) {
        return JinjaRenderer.renderWithSpans(template, context).text();
    }

    @Test
    void filterKeywordArgumentsAreHonoredOrRefused() {
        // Llama 3.x templates: tools | tojson(indent=4)
        assertEquals(
                "{\n    \"a\": 1,\n    \"b\": [\n        2,\n        3\n    ]\n}",
                render("{{ {'a': 1, 'b': [2, 3]} | tojson(indent=4) }}"));
        assertEquals("{\"a\": 1, \"b\": [2, 3]}", render("{{ {'a': 1, 'b': [2, 3]} | tojson }}"));
        assertEquals("[]", render("{{ [] | tojson(indent=2) }}"));
        assertEquals("x", render("{{ v | default('x', boolean=true) }}", nullValue()));
        // the transformers tojson passthrough the template corpus uses
        assertEquals(
                "{\"a\":[1,2]}", render("{{ {'a': [1, 2]} | tojson(separators=(',', ':')) }}"));
        assertEquals("{\"k\": \"é\"}", render("{{ {'k': 'é'} | tojson(ensure_ascii=False) }}"));
        assertEquals(
                "{\"k\": \"\\u00e9\"}", render("{{ {'k': 'é'} | tojson(ensure_ascii=True) }}"));
        assertEquals(
                "{\"a\": 1, \"b\": 2}", render("{{ {'b': 2, 'a': 1} | tojson(sort_keys=True) }}"));
        assertEquals(
                "{\n  \"a\": 1\n}",
                render("{{ {'a': 1} | tojson(indent=2, separators=(',', ': ')) }}"));
        RuntimeException e =
                assertThrows(
                        RuntimeException.class, () -> render("{{ 'a' | upper(strict=true) }}"));
        assertTrue(e.getMessage().contains("keyword argument 'strict'"), e.getMessage());
    }

    @Test
    void listsConcatenateAndIntegersStayIntegers() {
        // the namespace accumulate idiom: ns.msgs = ns.msgs + [m]
        assertEquals(
                "2",
                render(
                        "{% set ns = namespace(msgs=[]) %}{% for m in [1, 2] %}{% set ns.msgs ="
                                + " ns.msgs + [m] %}{% endfor %}{{ ns.msgs | length }}"));
        assertEquals("[1, 2, 3]", render("{{ [1] + [2, 3] }}"));
        assertEquals("3", render("{{ 1 + 2 }}"));
        assertEquals("3.5", render("{{ 1 + 2.5 }}"));
        assertEquals("6", render("{{ 2 * 3 }}"));
        assertEquals("2.0", render("{{ 4 / 2 }}"));
        assertEquals("1.5", render("{{ 5.5 % 2 }}"));
        assertEquals("2", render("{{ -1 % 3 }}"), "Python's sign rule");
        assertEquals("1.0", render("{{ 1.0 }}"));
        assertEquals("2.0", render("{{ 2.0 | tojson }}"));
        assertEquals("[1, 2.5]", render("{{ [1, 2.5] }}"));
    }

    @Test
    void dictLiteralsHaveTheDictMethods() {
        assertEquals(
                "a=1;", render("{% for k, v in {'a': 1}.items() %}{{ k }}={{ v }};{% endfor %}"));
        assertEquals("a", render("{{ {'a': 1}.keys() | join(',') }}"));
        assertEquals("1", render("{{ {'a': 1}.get('a') }}"));
        assertEquals("z", render("{{ {'a': 1}.get('b', 'z') }}"));
    }

    @Test
    void explicitDelimiterSplitPreservesAnEmptyString() {
        assertEquals("", render("{{ ''.split('<tool_sep>')[0].lstrip() }}"));
    }

    @Test
    void stringsIterateByCharacter() {
        assertEquals("[a][b]", render("{% for c in 'ab' %}[{{ c }}]{% endfor %}"));
        assertEquals("[é][😀]", render("{% for c in 'é😀' %}[{{ c }}]{% endfor %}"));
        assertEquals("", render("{% for x in none_value %}x{% endfor %}", nullValue()));
        assertEquals("", render("{% for x in undefined_value %}x{% endfor %}"));
    }

    @Test
    void forElseRunsOnAnEmptyIterableAndStrayKeywordsAreErrors() {
        assertEquals("b", render("{% for x in [] %}a{% else %}b{% endfor %}"));
        assertEquals("aa", render("{% for x in [1, 2] %}a{% else %}b{% endfor %}"));
        assertEquals(
                "b",
                render("{% for x in [1] if x > 5 %}a{% else %}b{% endfor %}"),
                "after the loop filter");
        for (String stray : List.of("{% endif %}", "x{% else %}y", "{% endfor %}")) {
            RuntimeException e = assertThrows(RuntimeException.class, () -> render(stray), stray);
            assertTrue(e.getMessage().contains("unexpected"), e.getMessage());
        }
    }

    @Test
    void defaultHonorsItsBooleanArgument() {
        assertEquals("None", render("{{ v | default('x') }}", nullValue()), "None is defined");
        assertEquals("x", render("{{ v | default('x', true) }}", nullValue()));
        assertEquals("x", render("{{ '' | default('x', true) }}"));
        assertEquals("x", render("{{ missing | default('x') }}"));
        assertEquals("keep", render("{{ 'keep' | default('x', true) }}"));
    }

    @Test
    void whitespaceControlStripsOnlyTheAdjacentText() {
        assertEquals("a \nb", render("a \n{{ x }}{%- if true %}b{%- endif %}", Map.of("x", "")));
        assertEquals("ab", render("a  {%- if true %}b{%- endif %}"));
        assertEquals("ab", render("a\n{{- 'b' }}"), "{{- strips the text right before it");
    }

    @Test
    void loopOutsideALoopIsUndefinedNotACrash() {
        assertEquals("no", render("{% if loop is defined %}yes{% else %}no{% endif %}"));
        assertEquals("1", render("{% for x in [7] %}{{ loop.index }}{% endfor %}"));
    }

    @Test
    void lengthOfNoneIsAnErrorAsInPython() {
        RuntimeException e =
                assertThrows(RuntimeException.class, () -> render("{{ v | length }}", nullValue()));
        assertTrue(e.getMessage().contains("len()"), e.getMessage());
        assertEquals("2", render("{{ 'ab' | length }}"));
        assertEquals("0", render("{{ [] | length }}"));
    }

    private static Map<String, Object> nullValue() {
        Map<String, Object> m = new HashMap<>();
        m.put("v", null);
        m.put("none_value", null);
        return m;
    }

    @Test
    void equalityFollowsPythonAcrossTypes() {
        assertEquals("True", render("{{ 1 == 1.0 }}"));
        assertEquals("True", render("{{ true == 1 }}"));
        assertEquals("False", render("{{ '1' == 1 }}"));
        assertEquals("True", render("{{ 'a' == 'a' }}"));
        assertEquals("False", render("{{ none == 0 }}"));
    }

    @Test
    void stringMethodsWorkWithParentheses() {
        assertEquals("a", render("{{ ' a '.strip() }}"));
    }

    @Test
    void andAndOrShortCircuitLikePython() {
        // the Nemotron templates guard every optional field this way; an eager right side
        // applied `length` to an undefined value and refused the whole render
        assertEquals("False", render("{{ x is defined and x | length > 0 }}"));
        assertEquals("ok", render("{{ (x is defined and x | length > 0) or 'ok' }}"));
        assertEquals("a", render("{{ 'a' or (x | length) }}"));
        assertEquals("True", render("{{ x is not defined or x | length > 0 }}"));
    }

    @Test
    void rawBlocksRenderVerbatim() {
        // jinja2 with HF's trim_blocks + lstrip_blocks: the newline after {% raw %} is body,
        // the one after {% endraw %} is trimmed, indentation before either tag is stripped
        assertEquals(
                "a\n {{ x }} {% if %}\nb",
                render("a{% raw %}\n {{ x }} {% if %}\n{% endraw %}\nb"));
        assertEquals("a\n  x    \nb", render("a\n  {% raw %}  x  {% endraw %}  \nb"));
        assertEquals("axb", render("a {%- raw -%}  x  {%- endraw -%} b"));
        assertEquals("xab\ny", render("x{% raw %}ab\n   {% endraw %}y"));
        assertEquals("ab\nz", render("{% raw %}ab\n  {% endraw %}{% if true %}z{% endif %}"));
        assertEquals("|q", render("{% raw %}{% endraw %}|{%raw%}q{%endraw%}"));
        RuntimeException e = assertThrows(RuntimeException.class, () -> render("{% raw %}a"));
        assertTrue(e.getMessage().contains("raw"), e.getMessage());
    }

    @Test
    void unknownStatementsAreRefused() {
        // each rendered as nothing, silently dropping what the template meant
        for (String source :
                new String[] {"{% endraw %}x", "{% endset %}", "{% endcall %}", "{% foo %}"}) {
            RuntimeException e =
                    assertThrows(
                            RuntimeException.class, () -> JinjaRenderer.template(source), source);
            assertTrue(e.getMessage().contains("unknown statement"), e.getMessage());
        }
    }

    @Test
    void loopControlsBreakAndContinue() {
        Map<String, Object> xs = Map.of("xs", List.of(1, 2, 3));
        assertEquals(
                "1",
                render(
                        "{% for x in xs %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endfor %}",
                        xs));
        assertEquals(
                "1133",
                render(
                        "{% for x in xs %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{{"
                                + " loop.index }}{% else %}E{% endfor %}",
                        xs));
        // break leaves only the innermost loop
        assertEquals(
                "11 |21 31 |",
                render(
                        "{% for x in xs %}{% for y in xs %}{% if y == 2 %}{% break %}{% endif"
                                + " %}{{ x }}{{ y }} {% endfor %}{% if x == 2 %}{% continue %}{%"
                                + " endif %}|{% endfor %}",
                        xs));
        // outside a loop: a loop's else, a macro body inside a loop, top level
        for (String source :
                new String[] {
                    "{% break %}",
                    "{% for x in xs %}{% else %}{% break %}{% endfor %}",
                    "{% for x in xs %}{% macro m() %}{% continue %}{% endmacro %}{% endfor %}",
                }) {
            RuntimeException e =
                    assertThrows(
                            RuntimeException.class, () -> JinjaRenderer.template(source), source);
            assertTrue(e.getMessage().contains("outside a loop"), e.getMessage());
        }
    }

    @Test
    void missingMacroParametersAreUndefined() {
        assertEquals(
                "False2True2",
                render(
                        "{% macro m(a, b=2) %}{{ a is defined }}{{ b }}{% endmacro %}{{ m() }}{{"
                                + " m(1) }}"));
        assertEquals(
                "[]False",
                render("{% macro m(a) %}[{{ a }}]{{ a is none }}{% endmacro %}{{ m() }}"));
        // an unbound parameter shadows a global of the same name, as in jinja2
        assertEquals(
                "False",
                render(
                        "{% macro m(messages) %}{{ messages is defined }}{% endmacro %}{{ m() }}",
                        Map.of("messages", List.of(1))));
    }

    @Test
    void stripFamilyTakesCharactersAndKeepsItsSide() {
        assertEquals(
                "  ab\n  |abxx|a |abc|xxab|ab",
                render(
                        "{{ '  ab\\n  '.rstrip('\\n ') ~ '\\n  ' }}|{{ 'xxabxx'.lstrip('x') }}|{{"
                                + " ' a '.lstrip() }}|{{ 'abc'.rstrip('') }}|{{"
                                + " 'xxabxx'.rstrip('x') }}|{{ 'xxabxx'.strip('x') }}"));
        // .rstrip('\n') must keep the indentation before the newline
        assertEquals("  ab\n  ", render("{{ '  ab\\n  \\n'.rstrip('\\n') }}"));
        assertEquals("a", render("{{ ' a '.strip(none) }}"));
    }

    @Test
    void popIsACallableWithPythonSemantics() {
        assertEquals(
                "3[1, 2]12[]",
                render(
                        "{% set ys = [1, 2, 3] %}{{ ys.pop() }}{{ ys }}{{ ys.pop(0) }}{{"
                                + " ys.pop(-1) }}{{ ys }}"));
        // naming the method mutates nothing
        assertEquals("[1, 2, 3]", render("{% set ys = [1, 2, 3] %}{% set f = ys.pop %}{{ ys }}"));
        RuntimeException e = assertThrows(RuntimeException.class, () -> render("{{ [].pop() }}"));
        assertTrue(e.getMessage().contains("pop"), e.getMessage());
    }

    @Test
    void floatsPrintLikePythonRepr() {
        assertEquals(
                "0.0005 1e+16 1.5e-05 1.2345678901234568e+17 1e+22 0.1 0.3333333333333333 0.0005",
                render(
                        "{{ 0.0005 }} {{ 10000000000000000.0 }} {{ 0.000015 }} {{"
                                + " 123456789012345678.0 }} {{ 10000000000000000000000.0 }} {{ 0.1"
                                + " }} {{ 1/3 }} {{ 0.00025 * 2 }}"));
        assertEquals(
                "2.0 -0.0 -1.5 100.0 1e-05",
                render("{{ 4/2 }} {{ -0.0 }} {{ -1.5 }} {{ 100.0 }} {{ 0.00001 }}"));
        assertEquals("[0.0005]", render("{{ [0.0005] | tojson }}"));
    }

    @Test
    void splitHonorsMaxsplit() {
        assertEquals(
                "['a', 'b,c'] ['a', 'b', 'c'] ['a', 'b '] ['a', '', 'b']",
                render(
                        "{{ 'a,b,c'.split(',', 1) }} {{ 'a b  c'.split() }} {{ ' a b '.split(none,"
                                + " 1) }} {{ 'a,,b'.split(',') }}"));
    }
}
