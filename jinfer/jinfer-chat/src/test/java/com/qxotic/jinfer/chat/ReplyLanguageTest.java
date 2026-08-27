package com.qxotic.jinfer.chat;

import static com.qxotic.jinfer.chat.ReplyLanguage.alt;
import static com.qxotic.jinfer.chat.ReplyLanguage.bytes;
import static com.qxotic.jinfer.chat.ReplyLanguage.call;
import static com.qxotic.jinfer.chat.ReplyLanguage.content;
import static com.qxotic.jinfer.chat.ReplyLanguage.free;
import static com.qxotic.jinfer.chat.ReplyLanguage.gbnf;
import static com.qxotic.jinfer.chat.ReplyLanguage.mark;
import static com.qxotic.jinfer.chat.ReplyLanguage.markId;
import static com.qxotic.jinfer.chat.ReplyLanguage.opt;
import static com.qxotic.jinfer.chat.ReplyLanguage.rep;
import static com.qxotic.jinfer.chat.ReplyLanguage.seq;
import static com.qxotic.jinfer.chat.ReplyLanguage.think;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.ReplyLanguage.Node;
import com.qxotic.jinfer.chat.ReplyLanguage.Selection;
import com.qxotic.jinfer.chat.ReplyLanguage.Walk;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * The region-tagged {@link ReplyLanguage.Walk} over synthetic families: one definition drives
 * parsing (the {@link ReplyParser} contract), constraining ({@code maskLogits} phases) and forcing
 * ({@code forcedPrefix}). The laws pinned here are the design's: the control rule ends the reply on
 * any unexpected control token (language-pinned NORMAL ids included), free holes stream unmasked
 * while authored scaffold never surfaces, calls are atomic with verbatim payload ids, same-opener
 * regions disambiguate through candidacy, close-less regions exit on payload acceptance, and every
 * structural ambiguity throws at build time.
 */
public final class ReplyLanguageTest {

    // ids 0..6: specials.  7, 8: the halves of a split UTF-8 code point.  9+: single chars.
    static final String[] SPECIALS = {
        "<think>", "</think>", "<call>", "</call>", "<end>", "<odd>", "[ARGS]"
    };
    static final String CHARS = "abxy{}\":1,()f2[] \n";
    static final int THINK = 0, END_THINK = 1, CALL = 2, END_CALL = 3, END = 4, ODD = 5, ARGS = 6;
    static final int HALF_1 = 7, HALF_2 = 8; // 0xC3, 0xA9: e-acute split across tokens

    static int ch(char c) {
        int at = CHARS.indexOf(c);
        if (at < 0) throw new IllegalArgumentException("no token for " + c);
        return SPECIALS.length + 2 + at;
    }

    static final Tokenizer TOK = new FakeTokenizer();

    /** One call named f, arguments = the raw captured payload - the per-tool parser shape. */
    static final Function<String, List<Content.ToolCall>> ONE =
            text -> List.of(new Content.ToolCall("", "get_weather", Map.of("raw", text.strip())));

    /** The synthetic family: think? content call* end - the eight-of-nine shape. */
    static Node family() {
        return seq(
                opt(think(mark("<think>"), free(), mark("</think>"))),
                content(free()),
                rep(weatherCall(), 0, -1),
                mark("<end>"));
    }

    static Node weatherCall() {
        return call(
                ONE,
                mark("<call>"),
                bytes("{"),
                gbnf("root ::= \"\\\"a\\\":\" (\"1\" | \"1,1\")"),
                bytes("}"),
                mark("</call>"));
    }

    static int[] toks(String chars) {
        int[] out = new int[chars.length()];
        for (int i = 0; i < chars.length(); i++) out[i] = ch(chars.charAt(i));
        return out;
    }

    record Step(String fragment, boolean reasoning) {}

    static List<Step> run(Walk w, int... tokens) {
        List<Step> steps = new ArrayList<>();
        for (int t : tokens) {
            ReplyParser.Fragment f = w.feed(t);
            if (!f.text().isEmpty()) steps.add(new Step(f.text(), w.reasoning()));
        }
        return steps;
    }

    static boolean[] admitted(Walk w) {
        int n = TOK.vocabulary().size();
        MemoryView<MemorySegment> logits =
                Views.allocateF32(MemoryAllocators.ofArena(Arena.ofAuto()), n);
        w.maskLogits(logits);
        float[] values = Views.toFloatArray(logits, "logits");
        boolean[] ok = new boolean[n];
        for (int i = 0; i < n; i++) ok[i] = values[i] == 0f;
        return ok;
    }

    @Test
    void whitespaceAfterAThinkCloseIsFramingNotContent() {
        Walk w = Selection.of(family(), TOK).walk();
        List<Step> steps = new ArrayList<>();
        steps.addAll(run(w, THINK));
        steps.addAll(run(w, toks("ab")));
        steps.addAll(run(w, END_THINK));
        steps.addAll(run(w, ch('\n'), ch('\n'), ch(' ')));
        steps.addAll(run(w, toks("xy")));
        steps.addAll(run(w, END));
        assertEquals(
                List.of(
                        new Step("a", true),
                        new Step("b", true),
                        new Step("x", false),
                        new Step("y", false)),
                steps,
                "the frame after </think> streams nothing");
        Message m = w.finish();
        assertEquals("xy", ((Content.Text) m.content().get(1)).text());
    }

    @Test
    void thinkContentCallFlowWithAtomicCallAndVerbatim() {
        Walk w = Selection.of(family(), TOK).walk();
        List<Step> steps = new ArrayList<>();
        steps.addAll(run(w, THINK));
        steps.addAll(run(w, toks("ab")));
        steps.addAll(run(w, END_THINK));
        steps.addAll(run(w, toks("xy")));
        assertFalse(w.ended());
        steps.addAll(run(w, CALL));
        steps.addAll(run(w, toks("{\"a\":1}")));
        steps.addAll(run(w, END_CALL));
        assertFalse(w.accepted(), "this language mandates its terminator");
        steps.addAll(run(w, END));
        assertTrue(w.accepted(), "the terminator is the accept boundary");
        assertEquals(
                List.of(
                        new Step("a", true),
                        new Step("b", true),
                        new Step("x", false),
                        new Step("y", false)),
                steps,
                "free holes stream by region; the call region and every mark stay silent");

        Message m = w.finish();
        assertEquals(3, m.content().size(), m.toString());
        assertEquals("ab", ((Content.Reasoning) m.content().get(0)).text());
        Content.Text text = (Content.Text) m.content().get(1);
        assertEquals("xy", text.text());
        assertEquals(List.of(ch('x'), ch('y')), text.verbatim().toList());
        Content.ToolCall c = (Content.ToolCall) m.content().get(2);
        assertEquals("get_weather", c.name());
        assertEquals(Map.of("raw", "{\"a\":1}"), c.arguments());
        assertEquals(
                IntSequence.of(toks("{\"a\":1}")).toList(),
                c.verbatim().toList(),
                "payload verbatim excludes the marks");
    }

    @Test
    void theControlRuleEndsTheReplyOnAnyUnexpectedSpecial() {
        Walk w = Selection.of(family(), TOK).walk();
        run(w, toks("x"));
        w.feed(ODD);
        assertTrue(w.ended(), "a control token no mark expects ends the reply");
        assertFalse(w.accepted(), "ended by violation at a non-accepting state is a cut");
        assertEquals("", w.feed(ch('y')).text(), "later feeds are inert");
        assertEquals("x", w.finish().text(), "committed content survives");
    }

    @Test
    void anUnexpectedSpecialMidThinkEndsButStillFlushesReasoning() {
        Walk w = Selection.of(family(), TOK).walk();
        run(w, THINK);
        run(w, toks("a"));
        w.feed(CALL); // not the think closer: the reply ends (calls-inside-think not authored)
        assertTrue(w.ended());
        assertEquals("a", ((Content.Reasoning) w.finish().content().get(0)).text());
    }

    @Test
    void maskPhasesFollowTheRegions() {
        // no content region: the structure must MASK to its openers; free holes pass through
        Node required = seq(rep(weatherCall(), 1, -1), mark("<end>"));
        Walk w = Selection.of(required, TOK).walk();
        boolean[] atStart = admitted(w);
        for (int t = 0; t < atStart.length; t++) {
            assertEquals(t == CALL, atStart[t], "structure dispatch admits only the call opener");
        }
        // inside the payload the segment grammar masks: after {"a": only '1' continues
        for (int t : new int[] {CALL, ch('{'), ch('"'), ch('a'), ch('"'), ch(':')}) w.feed(t);
        boolean[] inPayload = admitted(w);
        for (int t = 0; t < inPayload.length; t++) {
            assertEquals(t == ch('1'), inPayload[t], "the payload grammar masks token " + t);
        }
        // the family() tree opens with free-opening content: pass-through, nothing masked
        boolean[] freeOpen = admitted(Selection.of(family(), TOK).walk());
        for (boolean ok : freeOpen) assertTrue(ok, "a free-opening dispatch point is unmasked");
    }

    @Test
    void forcedPrefixDerivesTheSeedAndStopsAtTheFirstChoice() {
        Node required = seq(rep(weatherCall(), 1, -1), mark("<end>"));
        Selection sel = Selection.of(required, TOK);
        // <call> then the byte run up to the payload: extraction stops at the Gbnf boundary
        // (the payload's own fixed opening stays the grammar's job, not the seed's)
        int[] expected = {CALL, ch('{')};
        assertEquals(
                IntSequence.of(expected).toList(), IntSequence.of(sel.forcedPrefix()).toList());
        Walk w = sel.walk();
        for (int t : sel.forcedPrefix()) w.feed(t);
        assertFalse(w.ended());
        boolean[] ok = admitted(w);
        assertTrue(ok[ch('"')] && !ok[ch('a')] && !ok[ch('1')]);
    }

    @Test
    void closelessCallRegionExitsOnPayloadAcceptanceNotExhaustion() {
        // Mistral's shape with a whitespace-tolerant schema tail: after the balanced payload the
        // grammar ACCEPTS but is not exhausted (optional continuations pending) - the reply must
        // still be endable and the call must not be lost
        Node lang =
                seq(
                        content(free()),
                        opt(
                                call(
                                        text ->
                                                List.of(
                                                        new Content.ToolCall(
                                                                "",
                                                                "f",
                                                                Map.of("raw", text.strip()))),
                                        mark("<call>"),
                                        bytes("f"),
                                        mark("[ARGS]"),
                                        gbnf("root ::= \"(1)\" \",\"?"))),
                        mark("<end>"));
        Walk w = Selection.of(lang, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        run(w, toks("f"));
        run(w, ARGS);
        for (int t : toks("(1)")) w.feed(t);
        assertFalse(w.ended());
        w.feed(END); // accepting-but-not-exhausted payload: region exits, terminator consumed
        assertFalse(w.ended());
        assertTrue(w.accepted());
        Message m = w.finish();
        Content.ToolCall c = (Content.ToolCall) m.content().get(1);
        assertEquals("f", c.name());
        assertEquals(Map.of("raw", "f (1)"), c.arguments()); // [ARGS] = a word boundary
    }

    @Test
    void finishCommitsAnOpenAcceptingCloselessCall() {
        // generation ends (budget, driver stop) with the payload balanced but the region open:
        // a balanced close-less call at end of generation IS a call
        Node lang =
                seq(
                        content(free()),
                        opt(
                                call(
                                        text ->
                                                List.of(
                                                        new Content.ToolCall(
                                                                "",
                                                                "f",
                                                                Map.of("raw", text.strip()))),
                                        mark("<call>"),
                                        bytes("f"),
                                        mark("[ARGS]"),
                                        gbnf("root ::= \"(1)\" \",\"?"))),
                        mark("<end>"));
        Walk w = Selection.of(lang, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        run(w, toks("f"));
        run(w, ARGS);
        for (int t : toks("(1)")) w.feed(t);
        Message m = w.finish();
        assertEquals(2, m.content().size());
        assertEquals("f", ((Content.ToolCall) m.content().get(1)).name());
    }

    @Test
    void sameOpenerRegionsDisambiguateThroughCandidacy() {
        // the Harmony shape: two region kinds behind ONE opening mark, split by scaffold bytes
        Node harmonyish =
                seq(
                        rep(
                                alt(
                                        think(
                                                mark("<call>"),
                                                bytes("y"),
                                                mark("[ARGS]"),
                                                free(),
                                                mark("</call>")),
                                        call(
                                                ONE,
                                                mark("<call>"),
                                                bytes("f("),
                                                gbnf("root ::= \"1\""),
                                                bytes(")"),
                                                mark("</call>"))),
                                0,
                                -1),
                        mark("<end>"));
        // path A: the scaffold byte 'f' commits the CALL candidate
        Walk w = Selection.of(harmonyish, TOK).walk();
        run(w, CALL);
        assertNull(w.channel(), "candidacy is scaffold: no channel yet");
        run(w, toks("f("));
        run(w, toks("1)"));
        run(w, END_CALL);
        run(w, END);
        Message a = w.finish();
        assertEquals("get_weather", ((Content.ToolCall) a.content().get(0)).name());

        // path B: the scaffold byte 'y' commits the THINK candidate; 'y' itself never streams
        Walk v = Selection.of(harmonyish, TOK).walk();
        run(v, CALL);
        List<Step> steps = run(v, toks("y")); // authored scaffold: silent
        assertEquals(List.of(), steps);
        run(v, ARGS);
        steps = run(v, toks("ab"));
        assertEquals(List.of(new Step("a", true), new Step("b", true)), steps);
        run(v, END_CALL);
        run(v, END);
        assertEquals("ab", ((Content.Reasoning) v.finish().content().get(0)).text());
    }

    @Test
    void aSharedSpanRegionCarriesSeveralCalls() {
        // the LFM2 shape: one marker pair, several calls, comma-separated in one payload
        Function<String, List<Content.ToolCall>> both =
                text -> {
                    List<Content.ToolCall> out = new ArrayList<>();
                    String body = text.strip();
                    for (String piece : body.substring(1, body.length() - 1).split(",")) {
                        out.add(
                                new Content.ToolCall(
                                        "", piece.substring(0, 1), Map.of("raw", piece)));
                    }
                    return out;
                };
        Node lang =
                seq(
                        content(free()),
                        opt(
                                call(
                                        both,
                                        mark("<call>"),
                                        bytes("["),
                                        gbnf("root ::= \"f(1)\" (\",\" \"f(2)\")?"),
                                        bytes("]"),
                                        mark("</call>"))),
                        mark("<end>"));
        Walk w = Selection.of(lang, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        for (int t : toks("[f(1),f(2)]")) w.feed(t);
        run(w, END_CALL);
        run(w, END);
        Message m = w.finish();
        assertEquals(3, m.content().size());
        assertEquals("f", ((Content.ToolCall) m.content().get(1)).name());
        assertEquals(Map.of("raw", "f(2)"), ((Content.ToolCall) m.content().get(2)).arguments());
    }

    @Test
    void aDanglingPartialCodePointNeverPoisonsTheNextRegion() {
        Walk w = Selection.of(family(), TOK).walk();
        run(w, THINK);
        w.feed(ch('a'));
        w.feed(HALF_1); // the first byte of a two-byte code point, never completed
        run(w, END_THINK);
        List<Step> steps = run(w, toks("xy"));
        assertEquals(
                List.of(new Step("x", false), new Step("y", false)),
                steps,
                "content after the think region streams cleanly");
        run(w, END);
        Message m = w.finish();
        assertTrue(
                ((Content.Reasoning) m.content().get(0)).text().startsWith("a"),
                "the think text keeps its completed code points");
        assertEquals("xy", m.text());
    }

    @Test
    void aPinnedNormalIdIsControlEverywhere() {
        // the Gemma4 <eos> case: a terminator whose token the GGUF mistypes as NORMAL
        Node lang = seq(content(free()), markId("<eos>", ch('2')));
        Walk w = Selection.of(lang, TOK).walk();
        List<Step> steps = run(w, toks("x"));
        assertEquals(List.of(new Step("x", false)), steps);
        w.feed(ch('2')); // pinned: control, and here the expected terminator
        assertTrue(w.accepted());
        assertEquals("x", w.finish().text(), "the pinned id never streams as the text '2'");

        // and where it is NOT expected, the control rule ends the reply
        Node think =
                seq(
                        think(mark("<think>"), free(), mark("</think>")),
                        content(free()),
                        markId("<eos>", ch('2')));
        Walk v = Selection.of(think, TOK).walk();
        run(v, THINK);
        run(v, toks("a"));
        v.feed(ch('2'));
        assertTrue(v.ended(), "a pinned control id mid-hole is a boundary, not text");
    }

    @Test
    void aMarkerPairCallSpanClaimsInteriorControlTokensAsPayload() {
        // MiniCPM5's </param> closers and Gemma's quote token: the payload SYNTAX is specials,
        // and the parser must receive their spellings exactly as the old span parsers fed them
        Node spanFamily =
                seq(
                        content(free()),
                        rep(call(ONE, mark("<call>"), free(), mark("</call>")), 0, -1),
                        mark("<end>"));
        Walk w = Selection.of(spanFamily, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        w.feed(ch('{'));
        w.feed(ODD); // an interior special: payload, not a boundary
        w.feed(ch('}'));
        run(w, END_CALL);
        assertFalse(w.ended(), "interior specials never end a claimed span");
        run(w, END);
        Content.ToolCall c = (Content.ToolCall) w.finish().content().get(1);
        assertEquals(Map.of("raw", "{<odd>}"), c.arguments(), "the spelling reaches the parser");
        assertEquals(
                List.of(ch('{'), ODD, ch('}')),
                c.verbatim().toList(),
                "the interior special's id rides the verbatim");
    }

    @Test
    void aReopenedCallSpanSelfCloses() {
        // the old chained-span behavior: a second opener commits the first span (its partial
        // payload usually parses to no call) and starts the next
        Node spanFamily =
                seq(
                        content(free()),
                        rep(call(ONE, mark("<call>"), free(), mark("</call>")), 0, -1),
                        mark("<end>"));
        Walk w = Selection.of(spanFamily, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        w.feed(ch('{'));
        run(w, CALL); // re-open: self-close + re-enter
        assertFalse(w.ended());
        for (int t : toks("{\"a\":1}")) w.feed(t);
        run(w, END_CALL);
        run(w, END);
        Message m = w.finish();
        long calls = m.content().stream().filter(p -> p instanceof Content.ToolCall).count();
        assertEquals(2, calls, "both spans commit; the partial one is the parser's to judge");
    }

    @Test
    void finishDiscardsAnUnterminatedMarkerPairSpan() {
        // a span the generation never closed is no call - finish() must not commit the partial
        Node spanFamily =
                seq(
                        content(free()),
                        rep(call(ONE, mark("<call>"), free(), mark("</call>")), 0, -1),
                        mark("<end>"));
        Walk w = Selection.of(spanFamily, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        for (int t : toks("{\"a\"")) w.feed(t); // cut mid-payload, closer never arrives
        Message m = w.finish();
        assertTrue(
                m.content().stream().noneMatch(p -> p instanceof Content.ToolCall),
                "an unterminated marker-pair span commits nothing");
        assertEquals("x", m.text());
    }

    @Test
    void aByteBearingPinnedControlStillExitsAnAcceptingCloselessPayload() {
        // the rescue must read accepting() BEFORE tryAdvance walks the pinned token's bytes
        // into the payload grammar and kills the cursor (the mistyped-special class)
        Node lang =
                seq(
                        content(free()),
                        opt(
                                call(
                                        text -> List.of(new Content.ToolCall("", "f", Map.of())),
                                        mark("<call>"),
                                        bytes("f"),
                                        mark("[ARGS]"),
                                        gbnf("root ::= \"(1)\" \",\"?"))),
                        markId("<pin>", ch('2'))); // a NORMAL byte token pinned as the terminator
        Walk w = Selection.of(lang, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        run(w, toks("f"));
        run(w, ARGS);
        for (int t : toks("(1)")) w.feed(t);
        w.feed(ch('2')); // pinned control with REAL bytes: must exit + consume the terminator
        assertFalse(w.ended());
        assertTrue(w.accepted());
        assertEquals(
                1,
                w.finish().content().stream().filter(p -> p instanceof Content.ToolCall).count());
    }

    @Test
    void aCallRegionReportsTheToolCallChannel() {
        // ReplyLanes' unclaimed-call resurfacing keys on "tool-call" (both old parsers spoke it)
        Node spanFamily =
                seq(
                        content(free()),
                        rep(call(ONE, mark("<call>"), free(), mark("</call>")), 0, -1),
                        mark("<end>"));
        Walk w = Selection.of(spanFamily, TOK).walk();
        run(w, toks("x"));
        run(w, CALL);
        w.feed(ch('{'));
        assertEquals(Channel.TOOL_CALL, w.channel());
        assertFalse(w.outputChannels().contains(Channel.TOOL_CALL), "never an output channel");
    }

    @Test
    void streamedAndOneShotAgree() {
        IntSequence.Builder reply = IntSequence.newBuilder();
        reply.add(THINK);
        for (int t : toks("ab")) reply.add(t);
        reply.add(END_THINK);
        for (int t : toks("xy")) reply.add(t);
        reply.add(CALL);
        for (int t : toks("{\"a\":1}")) reply.add(t);
        reply.add(END_CALL);
        reply.add(END);
        IntSequence tokens = reply.build();
        Walk streamed = Selection.of(family(), TOK).walk();
        tokens.forEachInt(streamed::feed);
        Message one = ReplyParser.parse(Selection.of(family(), TOK).walk(), tokens);
        assertEquals(streamed.finish().toString(), one.toString());
    }

    @Test
    void unresolvableMarksPruneTheirAlternativesOrRejectTheSelection() {
        Node degradable =
                seq(alt(call(ONE, mark("<missing|>"), bytes("f")), content(free())), mark("<end>"));
        Walk w = Selection.of(degradable, TOK).walk(); // the call alternative pruned away
        run(w, toks("x"));
        w.feed(END);
        assertEquals("x", w.finish().text());

        Node unservable = seq(call(ONE, mark("<missing|>"), bytes("f")));
        assertThrows(UnsupportedOperationException.class, () -> Selection.of(unservable, TOK));
    }

    @Test
    void authoringErrorsThrowAtSelectionNeverMidGeneration() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Selection.of(seq(bytes("x")), TOK),
                "bytes at structure level");
        assertThrows(
                IllegalStateException.class,
                () -> Selection.of(seq(content(free()), content(free()), mark("<end>")), TOK),
                "two consecutive free-opening regions");
        assertThrows(
                IllegalStateException.class,
                () ->
                        Selection.of(
                                seq(
                                        think(mark("<think>"), free(), mark("</think>")),
                                        alt(content(free()), content(free())),
                                        mark("<end>")),
                                TOK),
                "an ambiguity DEEP in the tree still throws at build, not on a live request");
    }

    @Test
    void seedDropsScaffoldTextButKeepsTheParseState() {
        // a non-thinking scaffold seed (<think>a</think>) is PROMPT bytes: after seed the
        // finished reply carries none of it, and the streamed fragments equal the finished text
        Walk w = Selection.of(family(), TOK).walk();
        w.seed(IntSequence.of(THINK, ch('a'), END_THINK));
        StringBuilder streamed = new StringBuilder();
        for (int t : new int[] {ch('x'), ch('y'), END}) streamed.append(w.feed(t).text());
        Message m = w.finish();
        assertEquals("xy", m.text(), "seed scaffold must not lead the reply");
        assertEquals(streamed.toString(), m.text(), "streamed and finished must agree");
        assertTrue(
                m.content().stream().noneMatch(p -> p instanceof Content.Reasoning),
                "the seed's think text is not the reply's reasoning: " + m.content());
    }

    @Test
    void seedKeepsAnOpenSpanAndAForcedCallCapture() {
        // a prompt-OPENED think span stays open (state kept, seed text dropped): generated think
        // text still lands on the reasoning lane
        Walk w = Selection.of(family(), TOK).walk();
        w.seed(IntSequence.of(THINK, ch('a')));
        w.feed(ch('b'));
        w.feed(END_THINK);
        run(w, ch('x'), END);
        Message open = w.finish();
        assertEquals("b", ((Content.Reasoning) open.content().get(0)).text(), "seed 'a' dropped");
        assertEquals("x", open.text());

        // the forced-call exception: an open call capture survives the prompt seed
        Walk f = Selection.of(family(), TOK).walk();
        f.seed(IntSequence.of(ch('x'), CALL, ch('{')));
        run(f, ch('"'), ch('a'), ch('"'), ch(':'), ch('1'), ch('}'), END_CALL, END);
        Message forced = f.finish();
        List<Content.ToolCall> calls =
                forced.content().stream()
                        .filter(p -> p instanceof Content.ToolCall)
                        .map(p -> (Content.ToolCall) p)
                        .toList();
        assertEquals(1, calls.size(), "the seeded call must still commit: " + forced.content());
        assertEquals(
                "{\"a\":1}",
                calls.get(0).arguments().get("raw"),
                "the seed's opening brace stays in the payload");
    }

    /** The tools+schema shape: content is a GBNF payload, calls stay the family's own. */
    static Node schemaFamily() {
        return seq(
                rep(
                        alt(
                                content(gbnf("root ::= \"{\" \"1\" (\",\" \"1\")? \"}\"")),
                                weatherCall()),
                        0,
                        -1),
                mark("<end>"));
    }

    @Test
    void aGbnfOpeningContentRegionConstrainsTheReplyAndStillAdmitsCalls() {
        Walk w = Selection.of(schemaFamily(), TOK).walk();
        boolean[] ok = admitted(w);
        assertTrue(ok[ch('{')], "the schema's entry token is admissible at dispatch");
        assertTrue(ok[CALL], "the call opener stays a first-class alternative");
        assertTrue(ok[END], "so does the terminator");
        assertFalse(ok[ch('x')], "a plain token outside the schema is unrepresentable");
        assertFalse(ok[ch('1')], "even schema bytes out of position");

        // content streams (a Gbnf payload is the MODEL'S text), the call commits atomically
        List<Step> steps = run(w, ch('{'), ch('1'), ch('}'));
        assertEquals("{1}", steps.stream().map(Step::fragment).reduce("", String::concat));
        run(w, CALL, ch('{'), ch('"'), ch('a'), ch('"'), ch(':'), ch('1'), ch('}'), END_CALL, END);
        Message m = w.finish();
        assertEquals("{1}", m.text());
        assertEquals(
                1,
                m.content().stream().filter(p -> p instanceof Content.ToolCall).count(),
                "the call still commits: " + m.content());
    }

    @Test
    void aStatedContentHoleAppearsAtMostOnce() {
        // the spans preset's composed form: one request has ONE answer - after the document
        // completes, a second one is unrepresentable while calls stay legal
        Walk w =
                Selection.of(
                                ReplyLanguage.spans(
                                        "<think>",
                                        "</think>",
                                        "<call>",
                                        "</call>",
                                        ONE,
                                        mark("<end>"),
                                        gbnf("root ::= \"{\" \"1\" \"}\"")),
                                TOK)
                        .walk();
        run(w, ch('{'), ch('1'), ch('}'));
        boolean[] ok = admitted(w);
        assertFalse(ok[ch('{')], "no second document");
        assertTrue(ok[CALL], "calls stay legal after the answer");
        assertTrue(ok[END]);
    }

    // ---- fixture -----------------------------------------------------------

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocab = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocab;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++) out.add(ch(text.charAt(i)));
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            int id = tokens.intAt(tokenStartIndex);
            if (id == HALF_1) out.put((byte) 0xC3);
            else if (id == HALF_2) out.put((byte) 0xA9);
            else if (id < SPECIALS.length) out.put(SPECIALS[id].getBytes(StandardCharsets.UTF_8));
            else {
                out.put(
                        String.valueOf(CHARS.charAt(id - SPECIALS.length - 2))
                                .getBytes(StandardCharsets.UTF_8));
            }
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return SPECIALS.length + 2 + CHARS.length();
        }

        @Override
        public String token(int id) {
            if (id < SPECIALS.length) return SPECIALS[id];
            if (id == HALF_1 || id == HALF_2) return "<byte>";
            return String.valueOf(CHARS.charAt(id - SPECIALS.length - 2));
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < SPECIALS.length; i++) {
                if (SPECIALS[i].equals(text)) return i;
            }
            int at = text.length() == 1 ? CHARS.indexOf(text.charAt(0)) : -1;
            if (at < 0) throw new NoSuchElementException(text);
            return SPECIALS.length + 2 + at;
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < size();
        }

        @Override
        public boolean contains(String text) {
            try {
                id(text);
                return true;
            } catch (NoSuchElementException e) {
                return false;
            }
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            boolean special = id < SPECIALS.length;
            if (type == StandardTokenType.NORMAL) return !special;
            if (type == StandardTokenType.CONTROL) return special;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return Collections.emptyIterator();
        }
    }
}
