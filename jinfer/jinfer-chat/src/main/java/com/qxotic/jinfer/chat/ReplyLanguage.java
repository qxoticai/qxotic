package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.Function;

/**
 * A family's reply structure as ONE definition - a tree of {@link Node}s over the model's full
 * token alphabet - from which the three faces derive: PARSE (the {@link Walk} implements {@link
 * ReplyParser}), CONSTRAIN (the same walk masks logits), and FORCE (regions with exactly one
 * admissible path emit rather than sample; the leading such run is the {@link
 * Selection#forcedPrefix} injected into the prompt). See {@code docs/reply-language.md}.
 *
 * <p>THE CONTROL RULE: {@link Node.Free} admits plain tokens only; every control token - a
 * vocabulary special OR an id the language pinned as a mark (a GGUF-mistyped special) - either
 * matches a {@link Node.Mark} the language expects at the current state, or ends the reply. Stop
 * tokens, turn-fabrication guards and handoff stops all derive from this one rule.
 *
 * <p>Structure level ({@link Node.Seq}/{@link Node.Alt}/{@link Node.Repeat} over {@link
 * Node.Region}s and terminator {@link Node.Mark}s) dispatches on ONE token: a control token selects
 * the branch whose region it opens, a plain token selects the (at most one) free-opening region.
 * Several regions MAY share an opening mark (Harmony's four {@code <|channel|>} messages): the walk
 * then runs their opening grammars as parallel CANDIDATES and commits when one survives - candidate
 * tokens are scaffold in every branch, so nothing is ever emitted speculatively.
 *
 * <p>Region bodies mix {@link Node.Bytes}, {@link Node.Mark}, {@link Node.Gbnf} and {@link
 * Node.Free}; everything between free holes compiles into one {@link Grammar} spec, so interior
 * marks (Harmony's {@code <|constrain|>}) and schema payloads are ordinary grammar. Text belongs to
 * the model: free holes stream, and a constrained segment streams only when it carries a {@link
 * Node.Gbnf} payload (a schema-bound response) - authored {@link Node.Bytes} scaffold (Gemma's
 * {@code thought\n} channel name) never surfaces as text.
 */
public final class ReplyLanguage {

    private ReplyLanguage() {}

    /** The region kinds - the event, budget and selection unit. */
    public enum Kind {
        THINK,
        CONTENT,
        CALL
    }

    /** The seven-node vocabulary (plus the byte-level {@link Gbnf} payload escape). */
    public sealed interface Node {
        record Seq(List<Node> parts) implements Node {
            public Seq {
                parts = List.copyOf(parts);
            }
        }

        record Alt(List<Node> options) implements Node {
            public Alt {
                options = List.copyOf(options);
                if (options.isEmpty()) throw new IllegalArgumentException("empty alternation");
            }
        }

        /** {@code max} -1 = unbounded. */
        record Repeat(Node child, int min, int max) implements Node {
            public Repeat {
                if (min < 0 || max < -1 || (max >= 0 && max < min))
                    throw new IllegalArgumentException("bad repetition {" + min + "," + max + "}");
            }
        }

        /** Plain content bytes, tokenized plainly - inside region bodies only. */
        record Bytes(String literal) implements Node {
            public Bytes {
                if (literal == null || literal.isEmpty())
                    throw new IllegalArgumentException("empty byte literal");
            }
        }

        /**
         * ONE control token by IDENTITY. Resolved by {@code spelling} through the specials table at
         * selection; {@code pinned} >= 0 overrides for a token the container mistypes as NORMAL
         * (Gemma4's {@code <eos>}) - the walk treats a pinned id as control EVERYWHERE, free holes
         * included. A spelling absent from the vocabulary prunes every alternative containing it -
         * capability detection, derived.
         */
        record Mark(String spelling, int pinned) implements Node {
            public Mark {
                if (pinned < 0 && (spelling == null || spelling.isEmpty()))
                    throw new IllegalArgumentException("a mark needs a spelling or a pinned id");
            }
        }

        /** Pass-through: plain tokens stream, no mask; a control token is a boundary. */
        record Free() implements Node {}

        /** A byte-level GBNF payload (a schema grammar) - inside region bodies only. */
        record Gbnf(String source) implements Node {
            public Gbnf {
                if (source == null || source.isBlank())
                    throw new IllegalArgumentException("empty GBNF payload");
            }
        }

        /**
         * The event unit. {@code calls} is set on CALL regions exactly: it parses the region's
         * captured payload text - every non-mark byte in the region - into the calls it carries
         * (one for a per-tool body, several for a shared span like LFM2's bracket list; today's
         * span parsers have this exact shape). A single parsed call gets the payload's verbatim
         * ids; several parse with none (per-call attribution needs offsets nobody records).
         */
        record Region(Kind kind, Function<String, List<Content.ToolCall>> calls, List<Node> body)
                implements Node {
            public Region {
                body = List.copyOf(body);
                if (body.isEmpty()) throw new IllegalArgumentException("empty region body");
                if ((kind == Kind.CALL) != (calls != null))
                    throw new IllegalArgumentException("a calls parser on CALL regions exactly");
            }
        }
    }

    // ---- authoring factories ----------------------------------------------

    public static Node seq(Node... parts) {
        return new Node.Seq(List.of(parts));
    }

    public static Node alt(Node... options) {
        return new Node.Alt(List.of(options));
    }

    public static Node rep(Node child, int min, int max) {
        return new Node.Repeat(child, min, max);
    }

    public static Node opt(Node child) {
        return new Node.Repeat(child, 0, 1);
    }

    public static Node bytes(String literal) {
        return new Node.Bytes(literal);
    }

    public static Node mark(String spelling) {
        return new Node.Mark(spelling, -1);
    }

    public static Node markId(String spelling, int id) {
        return new Node.Mark(spelling, id);
    }

    public static Node free() {
        return new Node.Free();
    }

    public static Node gbnf(String source) {
        return new Node.Gbnf(source);
    }

    public static Node think(Node... body) {
        return new Node.Region(Kind.THINK, null, List.of(body));
    }

    public static Node content(Node... body) {
        return new Node.Region(Kind.CONTENT, null, List.of(body));
    }

    public static Node call(Function<String, List<Content.ToolCall>> calls, Node... body) {
        return new Node.Region(Kind.CALL, calls, List.of(body));
    }

    /**
     * The span-family preset, mirroring {@link ReplyParser#spans}: {@code think? (content |
     * call-span)* terminator?} - the reply shape of every marker-span family. Marks a vocabulary
     * lacks prune their alternatives (a checkpoint without call markers simply has no call syntax;
     * one without think markers has no reasoning span), so one preset serves every checkpoint of a
     * family.
     */
    public static Node spans(
            String thinkOpen,
            String thinkClose,
            String callOpen,
            String callClose,
            Function<String, List<Content.ToolCall>> calls,
            Node terminator) {
        return spans(thinkOpen, thinkClose, callOpen, callClose, calls, terminator, free());
    }

    /**
     * As {@link #spans(String, String, String, String, Function, Node)} with the CONTENT hole
     * stated: pass {@code gbnf(schemaGbnf)} and the family's visible text can only be that schema
     * while calls stay its own syntax - tools and a JSON response format as ONE selection.
     */
    public static Node spans(
            String thinkOpen,
            String thinkClose,
            String callOpen,
            String callClose,
            Function<String, List<Content.ToolCall>> calls,
            Node terminator,
            Node contentHole) {
        Node callSpan = call(calls, mark(callOpen), free(), mark(callClose));
        if (contentHole instanceof Node.Free) {
            return seq(
                    opt(think(mark(thinkOpen), free(), mark(thinkClose))),
                    rep(alt(content(contentHole), callSpan), 0, -1),
                    opt(terminator));
        }
        // a STATED hole appears at most once: one request has ONE answer, and a repeatable
        // schema region would admit a second document after the first completed
        return seq(
                opt(think(mark(thinkOpen), free(), mark(thinkClose))),
                rep(callSpan, 0, -1),
                opt(seq(content(contentHole), rep(callSpan, 0, -1))),
                opt(terminator));
    }

    /**
     * One span family's derived faces, held by its template - the marker spellings are written ONCE
     * and every {@link ChatTemplate} grammar word delegates here: {@code parser()} (the memoized
     * AUTO walk), {@code constrainedAuto} (the tree with the content hole stated - the tools +
     * JSON-schema seam) and {@code forcedCall} (headers forced through offered names). Pruning
     * still adapts the spellings per checkpoint.
     */
    public static final class Spans {
        private final String thinkOpen;
        private final String thinkClose;
        private final String callOpen;
        private final String callClose;
        private final Function<String, List<Content.ToolCall>> calls;
        private final Node terminator;
        private final Tokenizer tokenizer;
        private Selection auto; // memoized: tools-independent, built once

        public Spans(
                String thinkOpen,
                String thinkClose,
                String callOpen,
                String callClose,
                Function<String, List<Content.ToolCall>> calls,
                Node terminator,
                Tokenizer tokenizer) {
            this.thinkOpen = thinkOpen;
            this.thinkClose = thinkClose;
            this.callOpen = callOpen;
            this.callClose = callClose;
            this.calls = calls;
            this.terminator = terminator;
            this.tokenizer = tokenizer;
        }

        /** The family's memoized AUTO walk. */
        public Walk parser() {
            if (auto == null) auto = Selection.of(language(free()), tokenizer);
            return auto.walk();
        }

        /**
         * The compiled constrained selection - with tools, the composed shape (calls stay legal,
         * the answer optional); without, the document is REQUIRED and calls are out.
         */
        public Selection constrainedAuto(String contentGbnf, boolean toolsOffered) {
            if (toolsOffered) return Selection.of(language(gbnf(contentGbnf)), tokenizer);
            return Selection.of(
                    seq(
                            opt(think(mark(thinkOpen), free(), mark(thinkClose))),
                            content(gbnf(contentGbnf)),
                            opt(terminator)),
                    tokenizer);
        }

        /**
         * The compiled forced-call selection: per offered tool, the family's call region with the
         * header FORCED through the tool's name ({@code header} renders it, ending AT the name -
         * never the delimiter after it, whose merge the model was trained on) and the arguments the
         * model's own free span. Structure cannot derail: the walk holds the region to its closer
         * and the terminator, where a released pin used to hand the model an off-policy free
         * region.
         */
        public Selection forcedCall(List<Tool> tools, Function<Tool, String> header) {
            List<Node> options = new ArrayList<>(tools.size());
            for (Tool tool : tools) {
                options.add(
                        call(
                                calls,
                                mark(callOpen),
                                bytes(header.apply(tool)),
                                free(),
                                mark(callClose)));
            }
            return Selection.of(seq(new Node.Alt(options), opt(terminator)), tokenizer);
        }

        /** The family tree with the content hole stated. */
        private Node language(Node contentHole) {
            return spans(
                    thinkOpen, thinkClose, callOpen, callClose, calls, terminator, contentHole);
        }
    }

    // ---- selection: a concrete tree bound to one vocabulary ----------------

    /**
     * A tree bound to a tokenizer: marks resolved (unresolvable spellings prune their
     * alternatives), regions lowered to segments, the structure compiled to a dispatch program
     * validated WHOLE (every ambiguity throws at build, never mid-generation), the forced prefix
     * extracted. Throws {@link UnsupportedOperationException} when pruning leaves no live reply
     * (the request-time liveness law) and {@link IllegalArgumentException} or {@link
     * IllegalStateException} for authoring errors.
     */
    public static final class Selection {
        final Tokenizer tokenizer;
        final Op[] ops;
        final List<CRegion> regions;
        final Set<Integer> controlIds; // every resolved mark: pinned ids are control too
        final int entry;
        // per region: the entry-admissible token set of a GBNF-opening first segment (null for
        // mark- and free-opening regions) - structure masking's plain-dispatch union
        final long[][] regionEntry;
        private final int[] forcedPrefix;
        private Closure[] closures; // one per op, computed by validate - dispatch is a lookup

        public static Selection of(Node reply, Tokenizer tokenizer) {
            Node live =
                    prune(reply, tokenizer)
                            .orElseThrow(
                                    () ->
                                            new UnsupportedOperationException(
                                                    "no live reply: every alternative needs a"
                                                            + " control token this vocabulary"
                                                            + " does not have"));
            return new Selection(live, tokenizer);
        }

        private Selection(Node live, Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            Compiler c = new Compiler(tokenizer);
            this.entry = c.compile(live, c.op(Op.ACCEPT, 0, 0));
            this.ops = c.ops();
            this.regions = c.regions;
            this.controlIds = Set.copyOf(c.markIds);
            // a GBNF-OPENING region (tools + schema output: content is the schema payload)
            // dispatches on plain tokens its payload admits, so structure masking needs each
            // such region's ENTRY set - computed once here from a throwaway cursor, never from
            // the walk's own (the walk's cursor memo must stay untouched until entry)
            this.regionEntry = new long[regions.size()][];
            for (int i = 0; i < regions.size(); i++) {
                CRegion r = regions.get(i);
                if (r.opener() == -1 && r.segs().get(0) instanceof Seg.Spec s) {
                    regionEntry[i] = s.spec().cursor().admissible();
                }
            }
            validate();
            this.forcedPrefix = extractForcedPrefix();
        }

        /** Every op is a reachable walk position: ambiguity anywhere must throw NOW. */
        private void validate() {
            closures = new Closure[ops.length];
            for (int i = 0; i < ops.length; i++) {
                closures[i] = compute(i);
                if (ops[i].kind == Op.REGION) {
                    CRegion r = regions.get(ops[i].arg);
                    Closure after = closure(ops[i].next);
                    // FREE-opening only: a gbnf-opening region can exit on an inadmissible
                    // plain token at acceptance, so a plain-opening successor stays reachable
                    if (r.opener() == -1
                            && r.segs().get(0) instanceof Seg.Free
                            && after.plain() != -1
                            && after.marks().isEmpty()
                            && after.accept() == -1)
                        throw new IllegalStateException(
                                "two consecutive free-opening regions: the second is unreachable"
                                        + " (a free-final exit is always a control token) and"
                                        + " shields everything behind it");
                }
            }
        }

        public Walk walk() {
            return new Walk(this);
        }

        /**
         * The longest leading run with exactly one admissible path, canonically tokenized - the
         * derived forced seed / {@code replySeed}: inject into the prompt, then FEED to the walk so
         * parsing starts in the state the prompt left the model in. Extraction stops at the first
         * choice point; a payload's fixed opening stays the grammar's job.
         */
        public int[] forcedPrefix() {
            return forcedPrefix.clone();
        }

        private int[] extractForcedPrefix() {
            IntSequence.Builder out = IntSequence.newBuilder();
            Closure cl = closure(entry);
            while (cl.accept() == -1 && cl.plain() == -1 && cl.marks().size() == 1) {
                List<Integer> targets = cl.marks().values().iterator().next();
                if (targets.size() > 1) break; // a candidate group: the model chooses
                Op op = ops[targets.get(0)];
                if (op.kind != Op.REGION) break; // a bare terminator would end the reply
                if (!forcedRegion(regions.get(op.arg), out)) break;
                cl = closure(op.next);
            }
            return out.build().toArray();
        }

        /** Emits the region's tokens while single-path; true = the WHOLE region was forced. */
        private boolean forcedRegion(CRegion r, IntSequence.Builder out) {
            for (Seg seg : r.segs()) {
                if (!(seg instanceof Seg.Spec s)) return false; // a free hole: the model's turn
                for (int t : s.forced()) out.add(t);
                if (!s.fullyForced()) return false; // the run ends at the first choice
            }
            return true;
        }

        // -- mark pruning ----------------------------------------------------

        private static Optional<Node> prune(Node n, Tokenizer tok) {
            return switch (n) {
                case Node.Mark m -> resolve(m, tok).isPresent() ? Optional.of(n) : Optional.empty();
                case Node.Seq(List<Node> parts) -> pruneAll(parts, tok).map(Node.Seq::new);
                case Node.Alt(List<Node> options) -> {
                    List<Node> out = new ArrayList<>(options.size());
                    for (Node o : options) prune(o, tok).ifPresent(out::add);
                    yield out.isEmpty() ? Optional.empty() : Optional.of(new Node.Alt(out));
                }
                case Node.Repeat(Node child, int min, int max) ->
                        prune(child, tok)
                                .<Node>map(c -> new Node.Repeat(c, min, max))
                                .or(
                                        () ->
                                                min == 0
                                                        ? Optional.of(new Node.Seq(List.of()))
                                                        : Optional.empty());
                case Node.Region(Kind k, var calls, List<Node> body) ->
                        pruneAll(body, tok).map(b -> new Node.Region(k, calls, b));
                default -> Optional.of(n); // Bytes, Free, Gbnf
            };
        }

        /** All children live, or the whole construct is gone. */
        private static Optional<List<Node>> pruneAll(List<Node> nodes, Tokenizer tok) {
            List<Node> out = new ArrayList<>(nodes.size());
            for (Node n : nodes) {
                Optional<Node> kept = prune(n, tok);
                if (kept.isEmpty()) return Optional.empty();
                out.add(kept.get());
            }
            return Optional.of(out);
        }

        static OptionalInt resolve(Node.Mark m, Tokenizer tok) {
            return m.pinned() >= 0
                    ? OptionalInt.of(m.pinned())
                    : SpecialTokens.find(tok, m.spelling());
        }

        // -- closure over the dispatch program -------------------------------

        /**
         * What the structure admits here: opener id to target ops (several = a same-opener
         * CANDIDATE group, region targets only), at most one free-opening region, accept.
         */
        record Closure(Map<Integer, List<Integer>> marks, int plain, int accept) {}

        Closure closure(int at) {
            return closures != null && closures[at] != null ? closures[at] : compute(at);
        }

        private Closure compute(int at) {
            Map<Integer, List<Integer>> marks = new LinkedHashMap<>();
            int plain = -1, accept = -1;
            ArrayList<Integer> work = new ArrayList<>(List.of(at));
            for (int i = 0; i < work.size(); i++) {
                int op = work.get(i);
                switch (ops[op].kind) {
                    case Op.SPLIT -> {
                        work.add(ops[op].next);
                        work.add(ops[op].arg);
                    }
                    case Op.ACCEPT -> accept = op;
                    case Op.MARK -> put(marks, ops[op].arg, op, false);
                    case Op.REGION -> {
                        CRegion r = regions.get(ops[op].arg);
                        if (r.opener() == -1) {
                            if (plain != -1 && plain != op)
                                throw new IllegalStateException(
                                        "ambiguous structure: two free-opening regions");
                            plain = op;
                        } else {
                            put(marks, r.opener(), op, true);
                        }
                    }
                    default -> throw new IllegalStateException("op " + ops[op].kind);
                }
            }
            return new Closure(marks, plain, accept);
        }

        private void put(Map<Integer, List<Integer>> marks, int id, int op, boolean region) {
            List<Integer> targets = marks.computeIfAbsent(id, k -> new ArrayList<>());
            if (targets.contains(op)) return;
            // a shared opener is legal only as a REGION candidate group - a terminator mark
            // colliding with anything has no disambiguation
            if (!targets.isEmpty() && (!region || ops[targets.get(0)].kind != Op.REGION))
                throw new IllegalStateException(
                        "ambiguous structure: control token "
                                + id
                                + " opens both a terminator and something else");
            targets.add(op);
        }
    }

    // ---- compiled internals ------------------------------------------------

    /** A structure op: MARK consumes one control token, REGION runs a region, SPLIT branches. */
    static final class Op {
        static final byte MARK = 0, REGION = 1, SPLIT = 2, ACCEPT = 3;
        final byte kind;
        final int arg; // MARK: token id; REGION: region index; SPLIT: the other branch
        final int next;

        Op(byte kind, int arg, int next) {
            this.kind = kind;
            this.arg = arg;
            this.next = next;
        }
    }

    sealed interface Seg {
        /**
         * A region body between free holes, compiled to one grammar. {@code streams} = the segment
         * carries a {@link Node.Gbnf} payload, so its text is the MODEL'S (a schema-bound response)
         * and surfaces; scaffold segments stay silent.
         */
        record Spec(
                Grammar.Spec spec, int leadMark, int[] forced, boolean fullyForced, boolean streams)
                implements Seg {}

        /** A free hole; {@code closer} = the next segment's opening mark, -1 = region-final. */
        record Free(int closer) implements Seg {}
    }

    record CRegion(
            Kind kind,
            Function<String, List<Content.ToolCall>> calls,
            List<Seg> segs,
            Set<Integer> markIds,
            int opener,
            boolean spanShaped) {

        static CRegion of(
                Kind kind,
                Function<String, List<Content.ToolCall>> calls,
                List<Seg> segs,
                Set<Integer> markIds,
                boolean spanShaped) {
            // opener -1 covers TWO opening shapes sharing the plain-dispatch slot: a FREE hole
            // (pass-through) and a mark-less SPEC (a GBNF-opening region - the tools+schema
            // content shape - dispatching on exactly the plain tokens its payload admits)
            int opener =
                    segs.get(0) instanceof Seg.Spec s && s.leadMark() != -1 ? s.leadMark() : -1;
            return new CRegion(
                    kind, calls, List.copyOf(segs), Set.copyOf(markIds), opener, spanShaped);
        }
    }

    private static final class Compiler {
        private final Tokenizer tokenizer;
        private final List<Op> ops = new ArrayList<>();
        final List<CRegion> regions = new ArrayList<>();
        final Set<Integer> markIds = new LinkedHashSet<>();

        Compiler(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        Op[] ops() {
            return ops.toArray(new Op[0]);
        }

        int op(byte kind, int arg, int next) {
            ops.add(new Op(kind, arg, next));
            return ops.size() - 1;
        }

        /** Compiles a STRUCTURE node; returns its entry op. Continuation-passing, like the CFG. */
        int compile(Node n, int next) {
            return switch (n) {
                case Node.Seq(List<Node> parts) -> {
                    int c = next;
                    for (int i = parts.size() - 1; i >= 0; i--) c = compile(parts.get(i), c);
                    yield c;
                }
                case Node.Alt(List<Node> options) -> {
                    int c = compile(options.get(options.size() - 1), next);
                    for (int i = options.size() - 2; i >= 0; i--) {
                        c = op(Op.SPLIT, c, compile(options.get(i), next));
                    }
                    yield c;
                }
                case Node.Repeat(Node child, int min, int max) -> compileRep(child, min, max, next);
                case Node.Mark m -> {
                    int id = Selection.resolve(m, tokenizer).orElseThrow();
                    markIds.add(id);
                    yield op(Op.MARK, id, next);
                }
                case Node.Region r -> {
                    regions.add(lowerRegion(r));
                    yield op(Op.REGION, regions.size() - 1, next);
                }
                default ->
                        throw new IllegalArgumentException(
                                n.getClass().getSimpleName()
                                        + " cannot appear at structure level: bytes, payloads and"
                                        + " free holes live inside regions");
            };
        }

        private int compileRep(Node child, int min, int max, int next) {
            if (min == 0 && max == 1) return op(Op.SPLIT, next, compile(child, next));
            if (max == -1) {
                // star as a loop: a SPLIT whose taken branch re-enters and returns to itself
                int split = op(Op.SPLIT, 0, next); // arg patched to the body below
                int body = compile(child, split);
                ops.set(split, new Op(Op.SPLIT, body, next));
                int c = split;
                for (int i = 0; i < min; i++) c = compile(child, c);
                return c;
            }
            int c = next;
            for (int i = min; i < max; i++) c = op(Op.SPLIT, c, compile(child, c));
            for (int i = 0; i < min; i++) c = compile(child, c);
            return c;
        }

        // -- region lowering --------------------------------------------------

        private CRegion lowerRegion(Node.Region r) {
            List<Seg> segs = new ArrayList<>();
            List<Grammar.Term> run = new ArrayList<>();
            List<Integer> regionMarks = new ArrayList<>();
            boolean[] runStreams = {false};
            for (Node n : r.body()) {
                if (n instanceof Node.Free) {
                    if (!run.isEmpty()) {
                        segs.add(spec(run, runStreams[0]));
                        run = new ArrayList<>();
                        runStreams[0] = false;
                    }
                    segs.add(new Seg.Free(-1)); // closer patched below
                } else {
                    run.add(lowerBody(n, regionMarks, runStreams));
                }
            }
            if (!run.isEmpty()) segs.add(spec(run, runStreams[0]));
            for (int i = 0; i < segs.size(); i++) {
                if (!(segs.get(i) instanceof Seg.Free)) continue;
                int closer = -1;
                if (i + 1 < segs.size()) {
                    closer = ((Seg.Spec) segs.get(i + 1)).leadMark();
                    if (closer == -1)
                        throw new IllegalArgumentException(
                                "a free hole must be closed by a mark (or end the region)");
                }
                segs.set(i, new Seg.Free(closer));
            }
            markIds.addAll(regionMarks);
            return CRegion.of(
                    r.kind(), r.calls(), segs, new LinkedHashSet<>(regionMarks), spanShaped(r));
        }

        /**
         * Marks only as a leading and trailing run = a classic marker-pair span (SmolLM3), whose
         * whole payload is what the echo splices; interior marks (Harmony, Mistral) mean the span
         * ids are NOT the echo's splice unit.
         */
        private static boolean spanShaped(Node.Region r) {
            List<Node> body = r.body();
            int lead = 0;
            while (lead < body.size() && body.get(lead) instanceof Node.Mark) lead++;
            int tail = body.size();
            while (tail > lead && body.get(tail - 1) instanceof Node.Mark) tail--;
            for (int i = lead; i < tail; i++) {
                if (hasMark(body.get(i))) return false;
            }
            return true;
        }

        private static boolean hasMark(Node n) {
            return switch (n) {
                case Node.Mark m -> true;
                case Node.Seq(List<Node> parts) -> parts.stream().anyMatch(Compiler::hasMark);
                case Node.Alt(List<Node> options) -> options.stream().anyMatch(Compiler::hasMark);
                case Node.Repeat(Node child, int min, int max) -> hasMark(child);
                default -> false;
            };
        }

        private Seg.Spec spec(List<Grammar.Term> run, boolean streams) {
            Grammar.Term term = run.size() == 1 ? run.get(0) : new Grammar.Term.Seq(run);
            IntSequence.Builder forced = IntSequence.newBuilder();
            StringBuilder text = new StringBuilder();
            boolean whole = forcedInto(term, forced, text);
            flush(text, forced);
            return new Seg.Spec(
                    Grammar.of(term, tokenizer),
                    leadMark(term),
                    forced.build().toArray(),
                    whole,
                    streams);
        }

        /** Region-BODY node -> grammar term; collects mark ids and whether a payload appears. */
        private Grammar.Term lowerBody(Node n, List<Integer> marks, boolean[] streams) {
            return switch (n) {
                case Node.Bytes(String s) -> new Grammar.Term.Text(s);
                case Node.Gbnf(String src) -> {
                    streams[0] = true;
                    yield new Grammar.Term.Gbnf(src);
                }
                case Node.Mark m -> {
                    int id = Selection.resolve(m, tokenizer).orElseThrow();
                    marks.add(id);
                    yield new Grammar.Term.Token(id);
                }
                case Node.Seq(List<Node> parts) -> {
                    List<Grammar.Term> terms = new ArrayList<>(parts.size());
                    for (Node p : parts) terms.add(lowerBody(p, marks, streams));
                    yield new Grammar.Term.Seq(terms);
                }
                case Node.Alt(List<Node> options) -> {
                    List<Grammar.Term> terms = new ArrayList<>(options.size());
                    for (Node o : options) terms.add(lowerBody(o, marks, streams));
                    yield new Grammar.Term.Alt(terms);
                }
                case Node.Repeat(Node child, int min, int max) ->
                        new Grammar.Term.Rep(lowerBody(child, marks, streams), min, max);
                default ->
                        throw new IllegalArgumentException(
                                n.getClass().getSimpleName()
                                        + " cannot appear inside a region body (nested regions and"
                                        + " free holes under alt/rep are not a family shape)");
            };
        }

        private static int leadMark(Grammar.Term t) {
            return switch (t) {
                case Grammar.Term.Token(int id) -> id;
                case Grammar.Term.Seq(List<Grammar.Term> parts) ->
                        parts.isEmpty() ? -1 : leadMark(parts.get(0));
                default -> -1;
            };
        }

        /** Collects the term's single-path leading tokens; true = the whole term is forced. */
        private boolean forcedInto(Grammar.Term t, IntSequence.Builder out, StringBuilder text) {
            return switch (t) {
                case Grammar.Term.Text(String s) -> {
                    text.append(s);
                    yield true;
                }
                case Grammar.Term.Token(int id) -> {
                    flush(text, out);
                    out.add(id);
                    yield true;
                }
                case Grammar.Term.Seq(List<Grammar.Term> parts) -> {
                    for (Grammar.Term p : parts) {
                        if (!forcedInto(p, out, text)) yield false;
                    }
                    yield true;
                }
                default -> {
                    flush(text, out); // keep the run BEFORE the first choice point
                    yield false; // Alt, Rep, Gbnf: a choice exists
                }
            };
        }

        private void flush(StringBuilder text, IntSequence.Builder out) {
            if (text.isEmpty()) return;
            tokenizer.encode(text.toString()).forEachInt(out::add);
            text.setLength(0);
        }
    }

    // ---- the walk: parse + constrain + force, one object -------------------

    /**
     * One generation pass through a {@link Selection}: {@link #maskLogits} restricts sampling (free
     * holes pass through unmasked), {@link #feed} consumes the chosen token and returns the
     * displayable fragment - the {@link ReplyParser} contract, which this implements. The reply
     * ENDS ({@link #ended}) on the control rule: a control token nothing expects. Feed the {@link
     * Selection#forcedPrefix} first; single-use.
     */
    public static final class Walk implements ReplyParser {
        private final Selection sel;
        private final Tokenizer tokenizer;

        private int at; // structure op when between regions
        private CRegion region; // non-null while inside one
        private int seg;
        private Grammar.Cursor cursor; // the active Seg.Spec's
        private int regionReturn; // structure op after the open region
        // same-opener disambiguation: parallel first-segment cursors, commit on sole survivor
        private List<Cand> cands;
        private IntSequence.Builder candTokens;

        private record Cand(CRegion region, int ret, Grammar.Cursor cursor) {}

        private final PendingUtf8 pending = new PendingUtf8();
        private final ByteArrayOutputStream payload = new ByteArrayOutputStream();
        private IntSequence.Builder payloadIds = IntSequence.newBuilder();
        private IntSequence.Builder freeIds = IntSequence.newBuilder(); // CALL free holes only
        private final StringBuilder thinkText = new StringBuilder();
        private IntSequence.Builder thinkIds = IntSequence.newBuilder();
        private final StringBuilder contentText = new StringBuilder();
        private IntSequence.Builder contentIds = IntSequence.newBuilder();
        private final List<Content> calls = new ArrayList<>();
        private boolean lastReasoning;
        private boolean ended;
        private boolean generated;
        private boolean seeding;
        private Message finished;

        private Walk(Selection sel) {
            this.sel = sel;
            this.tokenizer = sel.tokenizer;
            this.at = sel.entry;
        }

        /**
         * True when the reply may legally end HERE by the language: at an accepting structure
         * point, or inside a region whose remaining obligation is a region-final free hole or an
         * accepting close-less payload. A reply ended by the control rule at a non-accepting state
         * was cut, not completed - this stays false there.
         */
        public boolean accepted() {
            if (cands != null) return false;
            if (region != null) {
                boolean atExit =
                        switch (region.segs().get(seg)) {
                            case Seg.Free f -> f.closer() == -1;
                            case Seg.Spec sp ->
                                    seg == region.segs().size() - 1
                                            && cursor != null
                                            && cursor.accepting();
                        };
                return atExit && sel.closure(regionReturn).accept() != -1;
            }
            return sel.closure(at).accept() != -1;
        }

        /** The control rule fired: a token arrived that nothing expects. Later feeds are inert. */
        @Override
        public boolean ended() {
            return ended;
        }

        /**
         * The walk as a decode driver - mask, sample, feed, the ONE way a selection constrains a
         * generation (the forced-call path and the tools+schema selection both ride it). A walk
         * with nothing admissible emits {@code endTurn} (the model's own end of turn) forever.
         */
        public Sampler sampler(Sampler base, int endTurn) {
            return logits -> {
                if (!maskLogits(logits)) return endTurn;
                int token = base.sampleToken(logits);
                feed(token);
                return token;
            };
        }

        /**
         * Masks {@code logits} to what the language admits here; returns false when nothing is
         * admissible (only after {@link #ended}). Free holes and free-opening dispatch points pass
         * through unmasked - AUTO content is the model's own. During candidacy the mask is the
         * UNION of the candidates' admissions.
         */
        public boolean maskLogits(MemoryView<?> logits) {
            if (ended) return false;
            if (cands != null) {
                MemoryView<MemorySegment> writable = writable(logits);
                int n = Math.toIntExact(logits.shape().size());
                long[] union = new long[(n + 63) >> 6];
                for (Cand c : cands) {
                    long[] m = c.cursor.admissible();
                    if (m == null) return true; // a DISABLED payload: pass-through
                    for (int i = 0; i < union.length && i < m.length; i++) union[i] |= m[i];
                }
                for (int t = 0; t < n; t++) {
                    if ((union[t >> 6] & (1L << (t & 63))) == 0) reject(writable, t);
                }
                return true;
            }
            if (region != null) {
                if (region.segs().get(seg) instanceof Seg.Spec spec)
                    return cursor(spec).maskLogits(logits);
                return true; // a free hole: pass-through
            }
            Selection.Closure cl = sel.closure(at);
            long[] plainEntry = null;
            if (cl.plain() != -1) {
                plainEntry = sel.regionEntry[sel.ops[cl.plain()].arg];
                if (plainEntry == null) return true; // a FREE-opening region: pass-through
                // a GBNF-opening region: the mask is the union of the payload's entry set, the
                // closure's marks, and (at an accept position) the control exits
            }
            MemoryView<MemorySegment> writable = writable(logits);
            int n = Math.toIntExact(logits.shape().size());
            boolean accept = cl.accept() != -1;
            for (int t = 0; t < n; t++) {
                boolean ok =
                        (plainEntry != null && (plainEntry[t >> 6] >>> (t & 63) & 1L) != 0)
                                || cl.marks().containsKey(t)
                                || (accept && control(t));
                if (!ok) reject(writable, t);
            }
            return true;
        }

        @Override
        public Fragment feed(int token) {
            if (finished != null) throw new IllegalStateException("parser already finished");
            if (!seeding) generated = true;
            if (ended) return Fragment.EMPTY;
            if (cands != null) return feedCandidates(token);
            return region != null ? feedRegion(token) : dispatch(token);
        }

        // -- structure dispatch ----------------------------------------------

        private Fragment dispatch(int token) {
            Selection.Closure cl = sel.closure(at);
            List<Integer> targets = cl.marks().get(token);
            if (targets != null) {
                if (targets.size() > 1) return enterCandidates(targets, token);
                Op op = sel.ops[targets.get(0)];
                if (op.kind == Op.MARK) { // a terminator: scaffold, inert
                    at = op.next;
                    return Fragment.EMPTY;
                }
                enter(sel.regions.get(op.arg), op.next);
                // the opening mark is consumed by the region's own first-segment grammar (its
                // Token term), not by dispatch
                return feedRegion(token);
            }
            if (!control(token) && cl.plain() != -1) {
                Op op = sel.ops[cl.plain()];
                enter(sel.regions.get(op.arg), op.next);
                return feedRegion(token);
            }
            ended = true; // the control rule: nothing here expects this token
            flushPending(null);
            return Fragment.EMPTY;
        }

        /** Control = a vocabulary special, an empty-byte token, or a language-pinned mark id. */
        private boolean control(int token) {
            return SpecialTokens.isSpecial(tokenizer, token)
                    || sel.controlIds.contains(token)
                    || tokenizer.decodeBytes(new int[] {token}).length == 0;
        }

        private void enter(CRegion r, int ret) {
            region = r;
            regionReturn = ret;
            seg = 0;
            cursor = null;
            payload.reset();
            payloadIds = IntSequence.newBuilder();
            freeIds = IntSequence.newBuilder();
        }

        // -- same-opener candidacy -------------------------------------------

        private Fragment enterCandidates(List<Integer> targets, int token) {
            cands = new ArrayList<>(targets.size());
            candTokens = IntSequence.newBuilder();
            for (int t : targets) {
                Op op = sel.ops[t];
                CRegion r = sel.regions.get(op.arg);
                cands.add(new Cand(r, op.next, ((Seg.Spec) r.segs().get(0)).spec().cursor()));
            }
            return feedCandidates(token);
        }

        private Fragment feedCandidates(int token) {
            List<Cand> alive = new ArrayList<>(cands.size());
            for (Cand c : cands) {
                if (c.cursor.tryAdvance(token)) alive.add(c);
            }
            if (alive.isEmpty()) {
                ended = true;
                flushPending(null);
                return Fragment.EMPTY;
            }
            candTokens.add(token);
            if (alive.size() > 1) {
                // a candidate finishing its opener while others still fit means the shared
                // segment does not disambiguate - an authoring error, caught here loudly
                for (Cand c : alive) {
                    if (c.cursor.exhausted())
                        throw new IllegalStateException(
                                "ambiguous same-opener regions: one opening grammar completed"
                                        + " while others still fit");
                }
                cands = alive;
                return Fragment.EMPTY;
            }
            Cand won = alive.get(0);
            IntSequence buffered = candTokens.build();
            cands = null;
            candTokens = null;
            enter(won.region(), won.ret());
            cursor = won.cursor(); // already advanced through the buffered tokens
            StringBuilder out = new StringBuilder();
            IntSequence.Builder outIds = IntSequence.newBuilder();
            Seg.Spec first = (Seg.Spec) region.segs().get(0);
            buffered.forEachInt(
                    t -> {
                        Fragment f = capture(t, first);
                        out.append(f.text());
                        f.tokens().forEachInt(outIds::add);
                    });
            if (cursor.exhausted()) {
                seg++;
                cursor = null;
                if (seg == region.segs().size()) exitRegion();
            }
            return out.isEmpty() ? Fragment.EMPTY : new Fragment(out.toString(), outIds.build());
        }

        // -- region walking ---------------------------------------------------

        private Fragment feedRegion(int token) {
            Seg s = region.segs().get(seg);
            if (s instanceof Seg.Free f) return feedFree(f, token);
            Seg.Spec spec = (Seg.Spec) s;
            // read BEFORE advancing: a byte-bearing control token (a pinned mistyped special)
            // would drive the cursor dead first and the rescue below would read a stale false
            boolean wasAccepting = cursor(spec).accepting();
            if (!cursor.tryAdvance(token)) {
                // an accepting close-less payload at the region's end: the region is COMPLETE,
                // the stray token is the structure's to judge (Mistral's </s> after the args)
                if (control(token) && seg == region.segs().size() - 1 && wasAccepting) {
                    exitRegion();
                    return dispatch(token);
                }
                ended = true; // off-language: an unexpected special, or dead bytes
                flushPending(region.kind());
                return Fragment.EMPTY;
            }
            Fragment fragment = capture(token, spec);
            if (cursor.exhausted()) {
                seg++;
                cursor = null;
                if (seg == region.segs().size()) exitRegion();
            }
            return fragment;
        }

        private Fragment feedFree(Seg.Free f, int token) {
            if (!control(token)) {
                byte[] bs = tokenizer.decodeBytes(new int[] {token});
                if (region.kind() == Kind.CALL) { // a call's free hole IS payload: atomic, silent
                    payload.writeBytes(bs);
                    payloadIds.add(token);
                    freeIds.add(token);
                    return Fragment.EMPTY;
                }
                return stream(bs, token);
            }
            if (f.closer() == token) { // the closing mark: consumed by the NEXT segment's grammar
                flushPending(region.kind());
                seg++;
                return feedRegion(token);
            }
            if (token == region.opener()) {
                if (region.kind() != Kind.CALL) {
                    return Fragment
                            .EMPTY; // a DUPLICATE opener mid-span is inert, as the span parsers had
                    // it
                    // (a prompt-opened think seed followed by the model's own <think>)
                }
                // a RE-OPENED call span self-closes the current one (the old chained-span
                // behavior): the partial payload goes to the parser - usually no call - and the
                // opener starts the next span
                exitRegion();
                return dispatch(token);
            }
            if (region.kind() == Kind.CALL && f.closer() != -1) {
                // a marker-pair call span claims EVERYTHING to its closer, interior control
                // tokens included AS THEIR SPELLINGS - MiniCPM5's </param> closers and Gemma's
                // <|"|> quote token are the payload syntax itself, and the payload parsers read
                // exactly the decoded text the old span parsers fed them
                payload.writeBytes(tokenizer.decodeBytes(new int[] {token}));
                payloadIds.add(token);
                freeIds.add(token);
                return Fragment.EMPTY;
            }
            if (f.closer() == -1) { // region-final free: exit, the structure decides
                exitRegion();
                return dispatch(token);
            }
            ended = true; // a control token mid-hole that is not the closer
            flushPending(region.kind());
            return Fragment.EMPTY;
        }

        private Grammar.Cursor cursor(Seg.Spec s) {
            if (cursor == null) cursor = s.spec().cursor();
            return cursor;
        }

        /**
         * Routes one consumed constrained token: marks are scaffold everywhere; CALL bytes
         * accumulate atomically; a streaming segment (a schema-bound payload) surfaces text;
         * scaffold segments stay silent.
         */
        private Fragment capture(int token, Seg.Spec spec) {
            if (region.markIds().contains(token)) {
                // a mark BETWEEN captured runs is a WORD BOUNDARY in a call payload (the old
                // parser's defense: "get_time<|constrain|>json" must not read "get_timejson");
                // leading marks (the span opener) separate nothing
                if (region.kind() == Kind.CALL && payload.size() > 0) payload.write(' ');
                return Fragment.EMPTY;
            }
            byte[] bs = tokenizer.decodeBytes(new int[] {token});
            if (bs.length == 0) return Fragment.EMPTY;
            if (region.kind() == Kind.CALL) {
                payload.writeBytes(bs);
                payloadIds.add(token);
                return Fragment.EMPTY; // calls are ATOMIC: nothing surfaces mid-region
            }
            if (!spec.streams())
                return Fragment.EMPTY; // authored scaffold (a channel name), never text
            return stream(bs, token);
        }

        /** UTF-8-safe streaming into the current region's lane. */
        private Fragment stream(byte[] bs, int token) {
            PendingUtf8.Fragment frag = pending.add(bs, token);
            if (frag == null) return Fragment.EMPTY;
            route(frag.text(), frag.ids());
            lastReasoning = region.kind() == Kind.THINK;
            return new Fragment(frag.text(), frag.ids());
        }

        private void route(String text, IntSequence ids) {
            if (region != null && region.kind() == Kind.THINK) {
                thinkText.append(text);
                ids.forEachInt(thinkIds::add);
            } else {
                contentText.append(text);
                ids.forEachInt(contentIds::add);
            }
        }

        @Override
        public void seed(IntSequence seed) {
            if (generated) throw new IllegalStateException("cannot seed after generated tokens");
            if (finished != null) throw new IllegalStateException("parser already finished");
            seeding = true;
            try {
                seed.forEachInt(this::feed);
            } finally {
                seeding = false;
            }
            // Prompt text is not reply text. Parse state and an open call capture survive.
            pending.flush();
            thinkText.setLength(0);
            thinkIds = IntSequence.newBuilder();
            contentText.setLength(0);
            contentIds = IntSequence.newBuilder();
        }

        private static MemoryView<MemorySegment> writable(MemoryView<?> logits) {
            Views.requireF32(logits, "logits");
            Views.requireContiguous(logits, "logits");
            MemoryView<MemorySegment> writable = Views.castToSegmentBacked(logits, "logits");
            Views.checkAlive(writable, "logits");
            return writable;
        }

        private static void reject(MemoryView<MemorySegment> logits, int token) {
            logits.memory()
                    .base()
                    .set(
                            ValueLayout.JAVA_FLOAT,
                            logits.byteOffset() + (long) token * Float.BYTES,
                            Float.NEGATIVE_INFINITY);
        }

        /** A dangling partial code point must never poison the next region's stream. */
        private void flushPending(Kind owner) {
            PendingUtf8.Fragment frag = pending.flush();
            if (frag == null || frag.text().isEmpty()) return;
            if (owner == Kind.THINK) {
                thinkText.append(frag.text());
                frag.ids().forEachInt(thinkIds::add);
            } else {
                contentText.append(frag.text());
                frag.ids().forEachInt(contentIds::add);
            }
        }

        private void exitRegion() {
            flushPending(region.kind());
            // think and content ACCUMULATE across regions: the reply's structure is one
            // coalesced reasoning part, one coalesced text part, then the calls (the ReplyParser
            // contract, and what every echo consumer expects)
            if (region.kind() == Kind.CALL) commitCall();
            region = null;
            at = regionReturn;
        }

        private void commitCall() {
            String text = new String(payload.toByteArray(), StandardCharsets.UTF_8);
            List<Content.ToolCall> parsed = region.calls().apply(text);
            // verbatim must cover EXACTLY what the echo splices back: the whole span for classic
            // marker-pair regions (SmolLM3's envelope), the free hole for header-shaped regions
            // (Harmony's args body), nothing when neither is exact - a wrong splice corrupts the
            // re-rendered wire, a missing one merely re-tokenizes
            IntSequence free = freeIds.build();
            IntSequence verbatim =
                    region.spanShaped() ? payloadIds.build() : free.length() > 0 ? free : null;
            if (parsed.size() == 1 && verbatim != null) {
                Content.ToolCall c = parsed.get(0);
                calls.add(new Content.ToolCall(c.id(), c.name(), c.arguments(), verbatim));
            } else {
                calls.addAll(parsed);
            }
            payload.reset();
            payloadIds = IntSequence.newBuilder();
            freeIds = IntSequence.newBuilder();
        }

        // -- the ReplyParser faces -------------------------------------------

        @Override
        public boolean reasoning() {
            return lastReasoning;
        }

        @Override
        public Channel channel() {
            if (ended || cands != null) return null;
            if (region == null) {
                // pre-entry to a free-opening region: a text token fed now joins content
                return sel.closure(at).plain() != -1 ? Channel.CONTENT : null;
            }
            if (region.kind() == Kind.CALL) return Channel.TOOL_CALL; // ReplyLanes keys on this
            boolean textual =
                    switch (region.segs().get(seg)) {
                        case Seg.Free f -> true;
                        case Seg.Spec sp -> sp.streams();
                    };
            if (!textual) return null;
            return region.kind() == Kind.THINK ? Channel.REASONING : Channel.CONTENT;
        }

        @Override
        public Channel pending() {
            // a grammar sequences its regions: at most one is open, none is ever suspended
            return null;
        }

        @Override
        public Set<Channel> outputChannels() {
            return Set.of(Channel.CONTENT);
        }

        @Override
        public Message finish() {
            if (finished != null) return finished;
            // an open free hole or accepting close-less payload flushes where it stood (an
            // unterminated think span is still reasoning; a balanced close-less call at the end
            // of generation IS a call - llama.cpp commits these too); an incomplete CALL payload
            // is discarded: a span the generation never closed is no call
            if (region != null) {
                boolean complete =
                        switch (region.segs().get(seg)) {
                            // a marker-pair span cut before its closer is NOT complete - a span
                            // the generation never closed is no call; only a region-final free
                            // (a close-less body) ends legitimately with the generation
                            case Seg.Free f -> f.closer() == -1;
                            case Seg.Spec sp ->
                                    seg == region.segs().size() - 1
                                            && cursor != null
                                            && cursor.accepting();
                        };
                if (complete && region.kind() == Kind.CALL) {
                    exitRegion();
                } else {
                    flushPending(region.kind());
                    region = null;
                }
            }
            List<Content> parts = new ArrayList<>();
            flushThink(parts);
            flushContent(parts);
            parts.addAll(calls);
            finished = new Message(Role.ASSISTANT, List.copyOf(parts));
            return finished;
        }

        private void flushThink(List<Content> parts) {
            if (thinkText.isEmpty()) return;
            IntSequence ids = thinkIds.build();
            parts.add(
                    new Content.Reasoning(
                            List.of(new Content.Text(thinkText.toString(), ids)), ids));
        }

        private void flushContent(List<Content> parts) {
            if (contentText.isEmpty()) return;
            parts.add(new Content.Text(contentText.toString(), contentIds.build()));
        }
    }
}
