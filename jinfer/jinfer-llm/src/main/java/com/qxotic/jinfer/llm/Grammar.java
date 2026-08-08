package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.*;
import com.qxotic.toknroll.Tokenizer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Grammar-constrained decoding: compile a grammar once, mask logits per token so the model can only
 * ever emit strings of the grammar's language.
 *
 * <p>A GBNF grammar is parsed into a {@link Rule} IR, then compiled into a {@link CFG} — a
 * byte-level <b>pushdown</b> grammar (a context-free matcher with an explicit stack). A {@link
 * Cursor} walks it token-by-token: {@link Cursor#maskLogits} restricts the logits to the tokens the
 * grammar can accept next, and {@link Cursor#advanceWith} consumes the chosen token's bytes.
 * Because the matcher carries a stack it represents arbitrarily nested / recursive grammars (real
 * JSON, balanced parens, …) correctly — a finite DFA cannot. Masks are computed once per distinct
 * matcher state and cached on the {@link Spec} (shared across cursors), so the per-token cost
 * amortises to a lookup — the same idea Outlines / XGrammar rely on.
 *
 * <h2>The GBNF dialect (llama.cpp compatible)</h2>
 *
 * <p>A grammar is a list of rules, one {@code name ::= body} per line. <b>The first-declared rule
 * is the start rule</b> — conventionally named {@code root}, but position decides, not the name. A
 * line without {@code ::=} continues the previous rule (alternatives may sit on their own lines);
 * {@code #} starts a comment to end of line (except inside a literal or class). Rule names are Java
 * identifiers plus hyphens ({@code kebab-case} works). Bodies compose:
 *
 * <ul>
 *   <li>{@code "literal"} — exact bytes; escapes {@code \" \\ \n \r \t \xNN}.
 *   <li>{@code [abc]} {@code [a-z0-9]} {@code [^,\n]} — byte classes: members, ranges, negation,
 *       the same escapes. {@code .} matches any byte.
 *   <li>{@code x | y} — alternation; {@code ( … )} — grouping.
 *   <li>{@code * + ?} and bounded {@code {m} {m,} {m,n}} — repetition of the preceding element
 *       ({@code "ab"{2,4}}, {@code [0-9]{1,3}}, {@code (num ",")*}).
 *   <li>{@code name} — a reference to another rule; recursion is fine (pushdown), an undefined
 *       reference throws at compile.
 * </ul>
 *
 * <h2>Recipes</h2>
 *
 * <pre>{@code
 * root ::= "yes" | "no"                        # closed answer set (or Grammar.choice)
 *
 * root ::= [0-9]{1,3} "." [0-9]{1,3} "." [0-9]{1,3}   # bounded numeric shapes (semver-ish)
 *
 * root ::= item ("," item)*                    # comma list, no trailing comma
 * item ::= [a-z]+
 *
 * root ::= ws obj                              # whitespace-tolerant JSON-ish framing
 * ws   ::= [ \t\n]{0,8}                        # ALWAYS cap ws: unbounded ws lets a reluctant
 * obj  ::= "{" ... "}"                         # model stall emitting whitespace forever
 * }</pre>
 *
 * <p>For JSON prefer the builders: {@link #json} (full RFC 8259), {@link #fromSchema} (exactly the
 * documents a JSON Schema admits), {@link #choice} (label sets) — all pre-bound to the same caps.
 * {@link #gbnfLiteral} escapes arbitrary text into a literal.
 *
 * <h2>Semantics and expectations</h2>
 *
 * <ul>
 *   <li><b>Byte-level:</b> classes and ranges are over BYTES, not code points. ASCII ranges work as
 *       written; a multi-byte range like {@code [а-я]} does not mean "Cyrillic letters" — spell
 *       non-ASCII alternatives as literals ({@code "α" | "β"}).
 *   <li><b>A token is admissible iff its whole byte string is accepted</b> from the current state;
 *       special tokens report empty bytes (control, not content) and are never admissible
 *       mid-grammar.
 *   <li><b>Dead ends end cleanly:</b> when no vocabulary token is admissible the driving sampler
 *       forces a stop token; a COMPLETE grammar (nothing may follow) ends the reply — choice
 *       grammars deliberately terminate the turn.
 *   <li><b>Caching:</b> specs cache per (source, vocabulary) — repeated compiles are free; masks
 *       cache per matcher state (capped, see {@code MASK_CACHE_CAP}).
 *   <li><b>Reasoning stays free:</b> driven through {@code RequestPolicy.constrained}, the grammar
 *       binds only the output channel — think spans sample unconstrained.
 * </ul>
 *
 * <h2>Known limitations</h2>
 *
 * <ul>
 *   <li>Left recursion is bounded best-effort ({@code CLOSURE_CAP}) — prefer right recursion
 *       ({@code list ::= item ("," list)?}).
 *   <li>No lookahead, no lazy quantifiers, no capture — this is a generator constraint, not a regex
 *       engine.
 *   <li>Pathologically ambiguous grammars hit the {@code MAX_STACKS} backstop.
 *   <li>Constrained output is on-language but not on-distribution-free: over-tight grammars at
 *       positions where the model wants something else degrade quality — leave the model room where
 *       the answer genuinely varies.
 * </ul>
 */
public final class Grammar {

    /** Cap on parallel stacks in a matcher state — a backstop against pathological grammars. */
    static final int MAX_STACKS = 1 << 14;

    /**
     * Cap on epsilon-closure size per step — bounds left-recursive grammars (best-effort) while
     * staying far above any non-left-recursive closure, which is tiny.
     */
    static final int CLOSURE_CAP = 1 << 13;

    /** Cap on cached masks per Spec, bounding memory for long-lived / deeply nested grammars. */
    static final int MASK_CACHE_CAP = 1 << 13;

    private static final Map<Tokenizer, Vocab> WRAPPERS =
            Collections.synchronizedMap(new WeakHashMap<>());
    private static final Map<Vocab, Map<String, Spec>> CACHES =
            Collections.synchronizedMap(new WeakHashMap<>());

    private Grammar() {}

    /**
     * The byte view of a vocabulary the compiler masks over: {@code bytes(id)} is what sampling id
     * appends to the text. The seam that lets tests (and non-toknroll callers) compile grammars
     * without a real tokenizer; specials must report EMPTY bytes (control, not content).
     */
    public interface Vocab {
        int size();

        byte[] bytes(int tokenId);
    }

    private static final byte[] NO_BYTES = new byte[0];

    static Vocab vocab(Tokenizer tok) {
        return WRAPPERS.computeIfAbsent(
                tok,
                t ->
                        new Vocab() {
                            public int size() {
                                return t.vocabulary().size();
                            }

                            public byte[] bytes(int id) {
                                // Specials are CONTROL, not content, whatever their literal
                                // rendering (LFM2's <|im_end|> decodes to its 10-char string).
                                // Empty bytes makes every special unsamplable mid-grammar and
                                // samplable exactly at accept states - the model can always END
                                // a completed constrained span with its natural stop token.
                                if (SpecialTokens.isSpecial(t, id)) return NO_BYTES;
                                return t.decodeBytes(new int[] {id});
                            }
                        });
    }

    private static Map<String, Spec> cache(Vocab v) {
        return CACHES.computeIfAbsent(
                v,
                k -> {
                    @SuppressWarnings("serial")
                    var m =
                            new LinkedHashMap<String, Spec>(16, 0.75f, true) {
                                @Override
                                protected boolean removeEldestEntry(Map.Entry<String, Spec> e) {
                                    return size() > 32;
                                }
                            };
                    return Collections.synchronizedMap(m);
                });
    }

    /** Full RFC 8259 JSON over the tokenizer's vocabulary (whitespace-tolerant, ws capped). */
    public static Spec json(Tokenizer t) {
        return json(vocab(t));
    }

    static Spec json(Vocab v) {
        // NUL-prefixed builtin keys can never collide with a user grammar string
        return cache(v).computeIfAbsent("\0json", k -> build(JSON_GRAMMAR, v));
    }

    /**
     * Minified JSON: the same language as {@link #json} but with no whitespace permitted anywhere
     * (no spaces/newlines between tokens, none at top level) — forces compact, token-efficient
     * output.
     */
    public static Spec jsonCompact(Tokenizer t) {
        return jsonCompact(vocab(t));
    }

    static Spec jsonCompact(Vocab v) {
        return cache(v).computeIfAbsent("\0jsonCompact", k -> build(JSON_COMPACT_GRAMMAR, v));
    }

    /**
     * Compiles a GBNF grammar (llama.cpp dialect) over the tokenizer's vocabulary. Specs cache per
     * (source, vocabulary): repeated compiles of the same grammar are free.
     */
    public static Spec of(String g, Tokenizer t) {
        return of(g, vocab(t));
    }

    public static Spec of(String g, Vocab v) {
        return cache(v).computeIfAbsent(g, k -> build(k, v));
    }

    static Spec build(String gbnf, Vocab v) {
        List<Rule> rules = parse(gbnf);
        if (rules.isEmpty()) return Spec.DISABLED;
        int vs = v.size();
        byte[][] tokenBytes = new byte[vs][];
        for (int t = 0; t < vs; t++) tokenBytes[t] = v.bytes(t);
        return new Spec(CFG.compile(rules), tokenBytes);
    }

    // ---- Compiled CFG (byte-level pushdown grammar) ------------------------
    //
    // The grammar is flattened into "slots". A slot is one of:
    //   TERM  — a 256-bit byte set + a continuation slot (the next slot after a matching byte)
    //   REF   — a rule id + a return slot (where to continue once that rule completes)
    //   END   — marks the end of a rule alternative (pop a frame)
    // A rule is a set of alternative entry slots. Groups and repetitions are desugared into
    // anonymous rules so every leaf is a TERM/REF/END — no nesting survives into the matcher.

    static final byte T_TERM = 0, T_REF = 1, T_END = 2;

    static final class CFG {
        final byte[] kind; // T_TERM | T_REF | T_END
        final int[] data; // TERM: terminal index;  REF: rule id;  END: unused
        final int[] next; // TERM/REF: continuation slot (>=0);    END: unused
        final long[][] terms; // terminal byte sets (256-bit, long[4]), indexed by TERM.data
        final int[][] alts; // alts[ruleId] = entry slots, one per alternative
        final int root; // root rule id

        CFG(byte[] kind, int[] data, int[] next, long[][] terms, int[][] alts, int root) {
            this.kind = kind;
            this.data = data;
            this.next = next;
            this.terms = terms;
            this.alts = alts;
            this.root = root;
        }

        boolean termHas(int ti, int b) {
            return (terms[ti][b >>> 6] & (1L << (b & 63))) != 0;
        }

        static CFG compile(List<Rule> rules) {
            return new Builder().run(rules);
        }

        // -- compiler ---------------------------------------------------------

        static final class Builder {
            byte[] kind = new byte[64];
            int[] data = new int[64];
            int[] next = new int[64];
            int n;
            final List<long[]> terms = new ArrayList<>();
            final List<int[]> alts = new ArrayList<>(); // index = rule id

            CFG run(List<Rule> rules) {
                for (int i = 0; i < rules.size(); i++) alts.add(null); // reserve named-rule ids
                for (Rule r : rules) alts.set(r.id, compileBody(r.body));
                return new CFG(
                        Arrays.copyOf(kind, n),
                        Arrays.copyOf(data, n),
                        Arrays.copyOf(next, n),
                        terms.toArray(new long[0][]),
                        alts.toArray(new int[0][]),
                        0);
            }

            int slot(byte k, int d, int nx) {
                if (n == kind.length) {
                    int g = n * 2;
                    kind = Arrays.copyOf(kind, g);
                    data = Arrays.copyOf(data, g);
                    next = Arrays.copyOf(next, g);
                }
                kind[n] = k;
                data[n] = d;
                next[n] = nx;
                return n++;
            }

            int term(long[] set, int cont) {
                int ti = terms.size();
                terms.add(set);
                return slot(T_TERM, ti, cont);
            }

            int ref(int rid, int cont) {
                return slot(T_REF, rid, cont);
            }

            int end() {
                return slot(T_END, -1, -1);
            }

            int newRule() {
                int id = alts.size();
                alts.add(null);
                return id;
            }

            /**
             * Compile a rule body (which may contain top-level {@code |}) into alternative entry
             * slots.
             */
            int[] compileBody(List<Rule.Element> body) {
                List<List<Rule.Element>> parts = splitAlts(body);
                int endSlot = end();
                int[] entries = new int[parts.size()];
                for (int i = 0; i < parts.size(); i++)
                    entries[i] = compileSeq(parts.get(i), endSlot);
                return entries;
            }

            int compileSeq(List<Rule.Element> elems, int cont) {
                int c = cont;
                for (int i = elems.size() - 1; i >= 0; i--) c = compileElem(elems.get(i), c);
                return c;
            }

            int compileElem(Rule.Element e, int cont) {
                return switch (e) {
                    case Rule.Element.Value(byte b) -> term(singleton(b), cont);
                    case Rule.Element.Dot ignored -> term(all(), cont);
                    case Rule.Element.CharClass(List<Byte> chars, boolean neg) ->
                            term(classSet(chars, neg), cont);
                    case Rule.Element.Ref(int rid) -> ref(rid, cont);
                    case Rule.Element.Group(List<Rule.Element> kids) -> {
                        boolean hasPipe =
                                kids.stream().anyMatch(k -> k instanceof Rule.Element.Pipe);
                        if (!hasPipe) yield compileSeq(kids, cont);
                        int rid = newRule();
                        alts.set(rid, compileBody(kids));
                        yield ref(rid, cont);
                    }
                    case Rule.Element.Repetition(Rule.Element child, int min, int max) ->
                            compileRep(child, min, max, cont);
                    case Rule.Element.Pipe ignored -> cont; // handled by splitAlts
                };
            }

            int compileRep(Rule.Element child, int min, int max, int cont) {
                if (min == 0 && max == 1) { // E?  :  R ::= E | ε
                    int rid = newRule(), endR = end();
                    int e1 = compileElem(child, endR);
                    alts.set(rid, new int[] {e1, endR});
                    return ref(rid, cont);
                }
                if (min == 0 && max < 0) { // E*  :  R ::= E R | ε
                    int rid = newRule(), endR = end();
                    int self = ref(rid, endR);
                    int e1 = compileElem(child, self);
                    alts.set(rid, new int[] {e1, endR});
                    return ref(rid, cont);
                }
                if (min == 1 && max < 0) { // E+  :  R ::= E R | E
                    int rid = newRule(), endR = end();
                    int self = ref(rid, endR);
                    int e1 = compileElem(child, self);
                    int e2 = compileElem(child, endR);
                    alts.set(rid, new int[] {e1, e2});
                    return ref(rid, cont);
                }
                // general E{min,max}: min mandatory copies, then a star (max<0) or optional copies
                int c = cont;
                if (max < 0) {
                    c = compileRep(child, 0, -1, c);
                    for (int i = 0; i < min; i++) c = compileElem(child, c);
                } else {
                    for (int i = min; i < max; i++) c = compileRep(child, 0, 1, c);
                    for (int i = 0; i < min; i++) c = compileElem(child, c);
                }
                return c;
            }

            static List<List<Rule.Element>> splitAlts(List<Rule.Element> body) {
                List<List<Rule.Element>> parts = new ArrayList<>();
                List<Rule.Element> cur = new ArrayList<>();
                for (Rule.Element e : body) {
                    if (e instanceof Rule.Element.Pipe) {
                        parts.add(cur);
                        cur = new ArrayList<>();
                    } else cur.add(e);
                }
                parts.add(cur);
                return parts;
            }

            static long[] singleton(byte b) {
                long[] m = new long[4];
                int x = b & 0xFF;
                m[x >>> 6] |= 1L << (x & 63);
                return m;
            }

            static long[] all() {
                return new long[] {-1L, -1L, -1L, -1L};
            }

            static long[] classSet(List<Byte> chars, boolean neg) {
                long[] m = new long[4];
                for (byte ch : chars) {
                    int x = ch & 0xFF;
                    m[x >>> 6] |= 1L << (x & 63);
                }
                if (neg) for (int i = 0; i < 4; i++) m[i] = ~m[i];
                return m;
            }
        }
    }

    // ---- Spec (compiled grammar + matcher) ---------------------------------

    /**
     * A compiled grammar plus its decoded token table; the matcher engine lives here so the
     * per-state mask cache can be shared across all cursors.
     *
     * <p>The per-token wiring, for callers driving sampling themselves:
     *
     * <pre>{@code
     * Grammar.Cursor cursor = spec.cursor();          // fresh per generation pass
     * if (!cursor.maskLogits(logits)) return stop;    // dead/complete: force a stop token
     * int token = sampler.sampleToken(logits);        // choose among admissible tokens
     * cursor.advanceWith(token);                      // consume its bytes
     * }</pre>
     *
     * <p>Most callers want the assembled path instead: {@code RequestPolicy.constrained} wires this
     * into the chat sampling stack (think spans free, output bound, dead-end stop).
     */
    public static final class Spec {
        static final Spec DISABLED = new Spec(null, null);

        final CFG cfg;
        final byte[][] tokenBytes;
        final State start;
        final Map<StateKey, long[]> maskCache = new ConcurrentHashMap<>();

        Spec(CFG cfg, byte[][] tokenBytes) {
            this.cfg = cfg;
            this.tokenBytes = tokenBytes;
            if (cfg == null) {
                start = null;
                return;
            }
            List<int[]> raws = new ArrayList<>();
            for (int e : cfg.alts[cfg.root]) raws.add(new int[] {e});
            start = expandSet(raws);
        }

        public Cursor cursor() {
            return new Cursor(this);
        }

        public boolean isValid() {
            return cfg != null;
        }

        // -- pushdown matcher -------------------------------------------------

        /**
         * Epsilon-closure of a set of raw stacks: follow REF/END epsilon moves (branching on rule
         * alternatives) until every surviving stack has a TERM on top, collecting those "ready"
         * stacks and whether any stack empties (accept). Iterative with an explicit worklist so
         * depth is heap- not call-stack-bound. {@code seen} (keyed on the whole stack) dedups the
         * closure: sequential refs to one rule yield distinct stacks and are both explored, while
         * left recursion yields ever-growing stacks, bounded by {@link #CLOSURE_CAP}.
         */
        private State expandSet(List<int[]> raws) {
            // Fast path (the dominant decode case): every stack already has a TERM on top, so
            // the closure is just dedup - no worklist, no seen-set, no StackKey allocations.
            boolean anyEps = false;
            for (int[] s : raws) {
                if (s.length == 0 || cfg.kind[s[s.length - 1]] != T_TERM) {
                    anyEps = true;
                    break;
                }
            }
            if (!anyEps) {
                if (raws.size() == 1) return new State(raws, false);
                List<int[]> ready = new ArrayList<>(raws.size());
                for (int[] s : raws) {
                    boolean dup = false;
                    for (int[] r : ready) {
                        if (Arrays.equals(r, s)) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup && ready.size() < MAX_STACKS) ready.add(s);
                }
                return new State(ready, false);
            }
            List<int[]> ready = new ArrayList<>();
            Set<StackKey> seen = new HashSet<>();
            boolean[] acc = {false};
            ArrayDeque<int[]> work = new ArrayDeque<>(raws);
            while (!work.isEmpty() && seen.size() <= CLOSURE_CAP) {
                int[] stack = work.poll();
                if (!seen.add(new StackKey(stack))) continue;
                if (stack.length == 0) {
                    acc[0] = true;
                    continue;
                }
                int top = stack[stack.length - 1];
                switch (cfg.kind[top]) {
                    case T_TERM -> {
                        if (ready.size() < MAX_STACKS) ready.add(stack);
                    }
                    case T_END -> work.add(Arrays.copyOf(stack, stack.length - 1));
                    default -> { // T_REF
                        int rid = cfg.data[top], ret = cfg.next[top];
                        // Tail-call elimination: a ref whose continuation is an END slot
                        // returns INTO a pop, so the frame and the pop cancel - skip it.
                        // Without this, right-recursive loops (E* / E+ compile to R ::= E R | ε)
                        // push one frame per iteration: stacks grow with the repetition length,
                        // automaton states never repeat (no mask-cache hits), masks go
                        // quadratic, and past CLOSURE_CAP the closure silently truncates.
                        int[] base =
                                ret < 0 || cfg.kind[ret] == T_END
                                        ? Arrays.copyOf(stack, stack.length - 1)
                                        : replaceLast(stack, ret);
                        for (int e : cfg.alts[rid]) work.add(append(base, e));
                    }
                }
            }
            return new State(ready, acc[0]);
        }

        /** One byte against the ready set → the raw (unexpanded) stacks that survive. */
        private List<int[]> step(List<int[]> ready, int b) {
            List<int[]> raws = new ArrayList<>();
            for (int[] s : ready) {
                int top = s[s.length - 1];
                if (cfg.termHas(cfg.data[top], b)) raws.add(replaceLast(s, cfg.next[top]));
            }
            return raws;
        }

        /**
         * Walk {@code len} bytes from a ready set; returns the resulting state, or null if the
         * bytes cannot be consumed (the grammar rejects them).
         */
        State walk(List<int[]> ready, byte[] bytes, int len) {
            List<int[]> cur = ready;
            State st = null;
            for (int i = 0; i < len; i++) {
                List<int[]> raws = step(cur, bytes[i] & 0xFF);
                if (raws.isEmpty()) return null;
                st = expandSet(raws);
                cur = st.ready;
            }
            return st;
        }

        long[] maskFor(List<int[]> ready, boolean accepting) {
            StateKey key = stateKey(ready, accepting);
            long[] m = maskCache.get(key);
            if (m != null) return m;
            m = computeMask(ready, accepting);
            if (maskCache.size() < MASK_CACHE_CAP) maskCache.putIfAbsent(key, m);
            return m;
        }

        private long[] computeMask(List<int[]> ready, boolean accepting) {
            int vocab = tokenBytes.length;
            long[] m = new long[(vocab + 63) >> 6];
            // First-byte filter: the union of all ready terminals. A token whose first byte is not
            // in it cannot match — reject in O(1) without a full walk (most of the vocab, in
            // practice).
            long[] firsts = new long[4];
            for (int[] s : ready) {
                long[] tm = cfg.terms[cfg.data[s[s.length - 1]]];
                for (int i = 0; i < 4; i++) firsts[i] |= tm[i];
            }
            for (int t = 0; t < vocab; t++) {
                byte[] bs = tokenBytes[t];
                boolean ok;
                if (bs.length == 0) ok = accepting;
                else {
                    int f = bs[0] & 0xFF;
                    ok =
                            (firsts[f >>> 6] & (1L << (f & 63))) != 0
                                    && walk(ready, bs, bs.length) != null;
                }
                if (ok) m[t >> 6] |= 1L << (t & 63);
            }
            return m;
        }

        private static StateKey stateKey(List<int[]> ready, boolean accepting) {
            // fast path: most matcher states are single-stack - skip the sort and the int[][]
            if (ready.size() == 1) {
                int[] s = ready.get(0);
                int[] flat = new int[s.length + 2];
                flat[0] = accepting ? 1 : 0;
                System.arraycopy(s, 0, flat, 1, s.length);
                flat[flat.length - 1] = -1;
                return new StateKey(flat);
            }
            int[][] arr = ready.toArray(new int[0][]);
            Arrays.sort(arr, Grammar::cmpIntArr);
            int total = 1;
            for (int[] s : arr) total += s.length + 1;
            int[] flat = new int[total];
            int p = 0;
            flat[p++] = accepting ? 1 : 0;
            for (int[] s : arr) {
                for (int x : s) flat[p++] = x;
                flat[p++] = -1;
            }
            return new StateKey(flat);
        }

        private static int[] replaceLast(int[] s, int v) {
            int[] c = s.clone();
            c[c.length - 1] = v;
            return c;
        }

        private static int[] append(int[] s, int v) {
            int[] c = Arrays.copyOf(s, s.length + 1);
            c[s.length] = v;
            return c;
        }
    }

    record State(List<int[]> ready, boolean accept) {}

    static final class StackKey {
        final int[] s;
        final int h;

        StackKey(int[] s) {
            this.s = s;
            this.h = Arrays.hashCode(s);
        }

        @Override
        public int hashCode() {
            return h;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof StackKey k && Arrays.equals(s, k.s);
        }
    }

    static final class StateKey {
        final int[] flat;
        final int h;

        StateKey(int[] flat) {
            this.flat = flat;
            this.h = Arrays.hashCode(flat);
        }

        @Override
        public int hashCode() {
            return h;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof StateKey k && Arrays.equals(flat, k.flat);
        }
    }

    static int cmpIntArr(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) if (a[i] != b[i]) return Integer.compare(a[i], b[i]);
        return Integer.compare(a.length, b.length);
    }

    // ---- Cursor ------------------------------------------------------------

    /**
     * The mutable walk of one generation through a {@link Spec}: {@code maskLogits} restricts the
     * next sample to admissible tokens, {@code advanceWith} consumes the chosen one. Single-use and
     * single-threaded - obtain a fresh {@code spec.cursor()} per generation pass.
     */
    public static final class Cursor {
        private final Spec spec;
        private List<int[]> ready;
        private boolean accepting;

        Cursor(Spec spec) {
            this.spec = spec;
            if (spec.cfg != null) {
                ready = spec.start.ready();
                accepting = spec.start.accept();
            }
        }

        /** Rewinds to the start state - equivalent to a fresh {@code spec.cursor()}. */
        public void reset() {
            if (spec.cfg == null) return;
            ready = spec.start.ready();
            accepting = spec.start.accept();
        }

        /**
         * True when the grammar is fully matched with NO continuation - nothing left to constrain.
         * A prefix grammar reaching this state has done its job; {@link Sampler#withPrefixGrammar}
         * then releases the mask (a dead state - no match, not accepting - is not exhausted).
         */
        public boolean exhausted() {
            return spec.cfg != null && accepting && ready.isEmpty();
        }

        /**
         * Masks {@code logits} to grammar-allowed tokens; returns whether any token remains. A
         * DISABLED spec is a pass-through (no masking, always true).
         */
        public boolean maskLogits(FloatTensor logits) {
            if (spec.cfg == null) return true;
            int vocab = spec.tokenBytes.length;
            if (vocab == 0) return false;
            long[] mask = spec.maskFor(ready, accepting);
            boolean any = false;
            for (int w = 0; w < mask.length; w++) {
                long bits = mask[w];
                if (bits == 0) { // 64 rejected: write without per-bit tests
                    int base = w << 6;
                    for (int b = 0; b < 64 && base + b < vocab; b++)
                        logits.setFloat(base + b, Float.NEGATIVE_INFINITY);
                    continue;
                }
                any = true; // this word admits something
                if (bits == -1L) continue; // 64 allowed: nothing to mask
                int base = w << 6;
                for (int b = 0; b < 64 && base + b < vocab; b++) {
                    if ((bits & (1L << b)) == 0) logits.setFloat(base + b, Float.NEGATIVE_INFINITY);
                }
            }
            return any;
        }

        /**
         * Consume a chosen token's bytes, advancing the grammar. Empty-byte tokens (EOS/control) do
         * not advance. An impossible token drives the cursor to a dead state.
         */
        public void advanceWith(int token) {
            if (spec.cfg == null || token < 0 || token >= spec.tokenBytes.length) return;
            byte[] bs = spec.tokenBytes[token];
            if (bs.length == 0) return;
            State st = spec.walk(ready, bs, bs.length);
            if (st == null) {
                ready = List.of();
                accepting = false;
            } else {
                ready = st.ready();
                accepting = st.accept();
            }
        }
    }

    // ---- JSON grammar ------------------------------------------------------
    //
    // Full, recursive JSON. Whitespace is optional (ws*) so compact output like {"a":1} is
    // accepted; nesting is handled by the pushdown matcher. root is a single value (objects,
    // arrays, and top-level scalars all allowed).

    // RFC 8259 / ECMA-404 compliant: surrounding whitespace at top level (ws value ws), and string
    // bodies exclude unescaped control chars (0x00-0x1F), which strict JSON forbids.
    static final String JSON_GRAMMAR =
            """
            root ::= ws value ws
            value ::= object | array | string | number | "true" | "false" | "null"
            object ::= "{" ws "}" | "{" ws string ws ":" ws value (ws "," ws string ws ":" ws value)* ws "}"
            array  ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
            string ::= "\\"" ([^"\\\\\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
            number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
            ws     ::= [ \\t\\n\\r]{0,8}
            """;

    // Minified JSON: same structure as JSON_GRAMMAR with every `ws` removed, so no whitespace is
    // accepted anywhere (compact output only).
    static final String JSON_COMPACT_GRAMMAR =
            """
            root ::= value
            value ::= object | array | string | number | "true" | "false" | "null"
            object ::= "{" "}" | "{" string ":" value ("," string ":" value)* "}"
            array  ::= "[" "]" | "[" value ("," value)* "]"
            string ::= "\\"" ([^"\\\\\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
            number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
            """;

    // ---- enum / choice -----------------------------------------------------

    /**
     * A grammar accepting exactly one of {@code options}, emitted as raw (unquoted) literals — e.g.
     * {@code choice(v, "yes", "no")} forces the model to answer yes or no.
     */
    static Spec choice(Vocab v, String... options) {
        StringBuilder sb = new StringBuilder("root ::= ");
        for (int i = 0; i < options.length; i++) {
            if (i > 0) sb.append(" | ");
            sb.append(gbnfLiteral(options[i]));
        }
        return of(sb.toString(), v);
    }

    /**
     * Exactly one of {@code options}, byte-literal - the closed-label-set (classification) gate.
     */
    public static Spec choice(Tokenizer t, String... options) {
        return choice(vocab(t), options);
    }

    // ---- JSON Schema -> grammar -------------------------------------------

    /**
     * Compiles a (common subset of) JSON Schema into a JSON-constrained grammar — typed structured
     * output, the way OpenAI's {@code json_schema} response-format and llama.cpp both work.
     *
     * <p>Supported: {@code type} (object, array, string, number, integer, boolean, null, or an
     * array of those), {@code properties} + {@code required}, {@code items}, {@code enum}, {@code
     * const}, and {@code anyOf}/{@code oneOf}. Object properties are emitted in the order of {@code
     * required} (or, when {@code required} is absent, all declared properties); other keywords
     * ({@code patternProperties}, {@code $ref}, numeric/length bounds, …) are ignored — the result
     * is always valid JSON satisfying the supported constraints, never a broken grammar.
     */
    static Spec fromSchema(Map<String, Object> schema, Vocab v) {
        return of(Schema.toGbnf(schema), v);
    }

    /**
     * A grammar admitting exactly the JSON documents valid under {@code schema} (parsed JSON Schema
     * map): types, required/optional properties, enums, nesting - no other keys, no other shapes.
     * Whitespace bounded so a reluctant model cannot stall on it.
     */
    public static Spec fromSchema(Map<String, Object> schema, Tokenizer t) {
        return fromSchema(schema, vocab(t));
    }

    /** Translates a JSON Schema node tree into a GBNF grammar string. */
    static final class Schema {
        private final StringBuilder rules = new StringBuilder();
        private int counter;

        static String toGbnf(Map<String, Object> schema) {
            Schema s = new Schema();
            // shared leaf rules (any-JSON fallbacks + scalars)
            // BOUNDED whitespace (llama.cpp-style): unbounded ws lets a reluctant model stall
            // forever without progress, growing a fresh matcher state (and a full-vocab mask
            // recompute) per whitespace token. Eight chars covers pretty-printing.
            s.rules.append("ws ::= [ \\t\\n\\r]{0,8}\n");
            s.rules.append(
                    "string ::= \"\\\"\" ([^\"\\\\\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\""
                            + " [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\"\n");
            s.rules.append("integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)\n");
            s.rules.append(
                    "number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]?"
                            + " [0-9]+)?\n");
            s.rules.append(
                    "value ::= jobject | jarray | string | number | \"true\" | \"false\" |"
                            + " \"null\"\n");
            s.rules.append(
                    "jobject ::= \"{\" ws (string ws \":\" ws value (ws \",\" ws string ws \":\" ws"
                            + " value)*)? ws \"}\"\n");
            s.rules.append("jarray ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n");
            String root = s.body(schema);
            return "root ::= ws (" + root + ") ws\n" + s.rules;
        }

        /** Allocate a named rule for {@code node} and return its name (for refs / recursion). */
        private String rule(Object node) {
            String name = "r" + (counter++);
            String b = body(node);
            rules.append(name).append(" ::= ").append(b).append("\n");
            return name;
        }

        @SuppressWarnings("unchecked")
        private String body(Object node) {
            if (!(node instanceof Map)) return "value";
            Map<String, Object> m = (Map<String, Object>) node;
            if (m.containsKey("const")) return gbnfLiteral(jsonEncode(m.get("const")));
            if (m.get("enum") instanceof List<?> en) return joinLiterals(en);
            Object union = m.containsKey("anyOf") ? m.get("anyOf") : m.get("oneOf");
            if (union instanceof List<?> subs) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < subs.size(); i++) {
                    if (i > 0) sb.append(" | ");
                    sb.append(rule(subs.get(i)));
                }
                return sb.length() == 0 ? "value" : sb.toString();
            }
            Object type = m.get("type");
            if (type instanceof List<?> types) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < types.size(); i++) {
                    if (i > 0) sb.append(" | ");
                    sb.append(typeBody(String.valueOf(types.get(i)), m));
                }
                return sb.length() == 0 ? "value" : sb.toString();
            }
            if (type instanceof String t) return typeBody(t, m);
            return "value";
        }

        private String typeBody(String type, Map<String, Object> m) {
            return switch (type) {
                case "object" -> objectBody(m);
                case "array" -> arrayBody(m);
                case "integer" -> "integer";
                case "number" -> "number";
                case "boolean" -> "(\"true\" | \"false\")";
                case "null" -> "\"null\"";
                case "string" -> "string";
                default -> "value";
            };
        }

        @SuppressWarnings("unchecked")
        private String objectBody(Map<String, Object> m) {
            Object propsObj = m.get("properties");
            if (!(propsObj instanceof Map) || ((Map<?, ?>) propsObj).isEmpty())
                return "\"{\" ws \"}\"";
            Map<String, Object> props = (Map<String, Object>) propsObj;
            List<String> keys = new ArrayList<>();
            // an EMPTY required list means "nothing mandatory", not "no properties" - fall back
            // to all declared properties (AiServices emits required=[] for extracted POJOs)
            if (m.get("required") instanceof List<?> req && !req.isEmpty()) {
                for (Object k : req)
                    if (props.containsKey(String.valueOf(k))) keys.add(String.valueOf(k));
            } else {
                keys.addAll(props.keySet());
            }
            if (keys.isEmpty()) return "\"{\" ws \"}\"";
            StringBuilder sb = new StringBuilder("\"{\" ws ");
            for (int i = 0; i < keys.size(); i++) {
                if (i > 0) sb.append(" ws \",\" ws ");
                String k = keys.get(i);
                sb.append(gbnfLiteral("\"" + jsonEsc(k) + "\""))
                        .append(" ws \":\" ws ")
                        .append(rule(props.get(k)));
            }
            return sb.append(" ws \"}\"").toString();
        }

        private String arrayBody(Map<String, Object> m) {
            String item = m.containsKey("items") ? rule(m.get("items")) : "value";
            return "\"[\" ws (" + item + " (ws \",\" ws " + item + ")*)? ws \"]\"";
        }

        private String joinLiterals(List<?> values) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) sb.append(" | ");
                sb.append(gbnfLiteral(jsonEncode(values.get(i))));
            }
            return sb.length() == 0 ? "value" : sb.toString();
        }
    }

    /** JSON-encode a scalar/array/object value to its on-the-wire form. */
    @SuppressWarnings("unchecked")
    static String jsonEncode(Object v) {
        if (v == null) return "null";
        if (v instanceof String s) return "\"" + jsonEsc(s) + "\"";
        if (v instanceof Boolean b) return b ? "true" : "false";
        if (v instanceof Number n) {
            if ((n instanceof Double || n instanceof Float)) {
                double d = n.doubleValue();
                if (!Double.isInfinite(d) && !Double.isNaN(d) && d == Math.rint(d))
                    return Long.toString((long) d);
                return Double.toString(d);
            }
            return n.toString();
        }
        if (v instanceof Map<?, ?> m) {
            StringBuilder sb = new StringBuilder("{");
            int i = 0;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (i++ > 0) sb.append(",");
                sb.append("\"")
                        .append(jsonEsc(String.valueOf(e.getKey())))
                        .append("\":")
                        .append(jsonEncode(e.getValue()));
            }
            return sb.append("}").toString();
        }
        if (v instanceof List<?> l) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(jsonEncode(l.get(i)));
            }
            return sb.append("]").toString();
        }
        return "\"" + jsonEsc(String.valueOf(v)) + "\"";
    }

    static String jsonEsc(String s) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
                }
            }
        }
        return b.toString();
    }

    /** Wrap raw bytes as a GBNF double-quoted literal that matches exactly those bytes. */
    public static String gbnfLiteral(String raw) {
        StringBuilder b = new StringBuilder("\"");
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) b.append("\\x").append(String.format("%02x", (int) c));
                    else b.append(c);
                }
            }
        }
        return b.append("\"").toString();
    }

    // ========================================================================
    // GBNF parser  (grammar text -> Rule IR)
    // ========================================================================

    static List<Rule> parse(String gbnf) {
        // Join continuation lines first: llama.cpp-style GBNF lets one rule span several
        // physical lines (alternatives on their own lines); a line without ::= belongs to the
        // rule above it. Splitting on raw newlines would silently DROP those alternatives.
        List<String> logical = new ArrayList<>();
        for (String raw : gbnf.split("\n")) {
            String line = stripComment(raw);
            if (line.trim().isEmpty()) continue;
            if (line.contains("::=") || logical.isEmpty()) logical.add(line);
            else logical.set(logical.size() - 1, logical.getLast() + " " + line);
        }
        Map<String, Integer> nameToId = new LinkedHashMap<>();
        List<Rule> rules = new ArrayList<>();
        // "root" is THE start symbol (llama.cpp's contract), so it takes id 0 - which is the id
        // the compiler starts from. Deriving the start symbol from declaration order instead
        // silently miscompiled every grammar whose helpers come first: `value ::= "a" | "b"` above
        // `root ::= "[" value "]"` matched `a`, not `[a]`, with no error anywhere.
        boolean declaresRoot = false;
        for (String line : logical) {
            int eq = line.indexOf("::=");
            if (eq >= 0 && line.substring(0, eq).trim().equals("root")) {
                declaresRoot = true;
                nameToId.put("root", 0);
                rules.add(null);
                break;
            }
        }
        for (String line : logical) {
            int eq = line.indexOf("::=");
            if (eq < 0) continue;
            String name = line.substring(0, eq).trim();
            if (!nameToId.containsKey(name)) {
                nameToId.put(name, rules.size());
                rules.add(null);
            }
        }
        if (!rules.isEmpty() && !declaresRoot) {
            throw new IllegalArgumentException(
                    "grammar has no 'root' rule - it is the start symbol, so there is nothing to"
                            + " match from; declared rules: "
                            + nameToId.keySet());
        }
        for (String line : logical) {
            int eq = line.indexOf("::=");
            if (eq < 0) continue;
            String name = line.substring(0, eq).trim();
            int id = nameToId.get(name);
            String body = line.substring(eq + 3).trim();
            rules.set(id, new Rule(id, parseBody(body, nameToId)));
        }
        for (int i = 0; i < rules.size(); i++)
            if (rules.get(i) == null) rules.set(i, new Rule(i, List.of()));
        return rules;
    }

    private static String stripComment(String line) {
        // '#' starts a comment unless inside a string literal OR a char class ([#] is a class
        // containing '#', not a comment)
        boolean inStr = false, escape = false, inClass = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inStr) {
                if (c == '\\') escape = !escape;
                else if (c == '"' && !escape) inStr = false;
                else escape = false;
            } else if (inClass) {
                if (c == '\\') escape = !escape;
                else if (c == ']' && !escape) inClass = false;
                else escape = false;
            } else if (c == '"') inStr = true;
            else if (c == '[') inClass = true;
            else if (c == '#') return line.substring(0, i);
        }
        return line;
    }

    private static List<Rule.Element> parseBody(String body, Map<String, Integer> rules) {
        List<Rule.Element> res = new ArrayList<>();
        int i = 0;
        while (i < body.length()) {
            char c = body.charAt(i);
            if (c == ' ' || c == '\t') {
                i++;
                continue;
            }
            if (c == '"') {
                int end = body.indexOf('"', i + 1);
                while (end > 0 && body.charAt(end - 1) == '\\') {
                    int slashes = 0, j = end - 1;
                    while (j >= 0 && body.charAt(j) == '\\') {
                        slashes++;
                        j--;
                    }
                    if (slashes % 2 == 0) break;
                    end = body.indexOf('"', end + 1);
                }
                if (end < 0) {
                    throw new IllegalArgumentException(
                            "unterminated string literal in rule body: " + body);
                }
                String s = unescape(body.substring(i + 1, end));
                for (byte b : s.getBytes(StandardCharsets.UTF_8))
                    res.add(new Rule.Element.Value(b));
                i = end + 1;
                i = applyMod(body, i, res);
            } else if (c == '[') {
                int end = findMatchingBracket(body, i);
                if (end < 0) {
                    i++;
                    continue;
                }
                res.add(parseCharClass(body.substring(i + 1, end)));
                i = end + 1;
                i = applyMod(body, i, res);
            } else if (c == '.') {
                res.add(new Rule.Element.Dot());
                i++;
                i = applyMod(body, i, res);
            } else if (c == '|') {
                res.add(new Rule.Element.Pipe());
                i++;
            } else if (Character.isJavaIdentifierStart(c)) {
                int end = i;
                while (end < body.length()
                        && (Character.isJavaIdentifierPart(body.charAt(end))
                                || body.charAt(end) == '-')) // llama.cpp names allow hyphens
                end++;
                String name = body.substring(i, end);
                Integer rid = rules.get(name);
                if (rid == null) {
                    throw new IllegalArgumentException(
                            "undefined rule reference '" + name + "' in rule body: " + body);
                }
                res.add(new Rule.Element.Ref(rid));
                i = applyMod(body, end, res);
            } else if (c == '(') {
                int end = findMatchingParen(body, i);
                if (end < 0) {
                    i++;
                    continue;
                }
                List<Rule.Element> inner = parseBody(body.substring(i + 1, end - 1), rules);
                res.add(new Rule.Element.Group(inner));
                i = applyMod(body, end, res);
            } else i++;
        }
        return res;
    }

    /**
     * The {@code [...]} char-class sub-parser: members, {@code a-z} ranges, {@code ^} negation, and
     * the literal escapes incl. {@code \\xNN} - all at the BYTE level.
     */
    private static Rule.Element.CharClass parseCharClass(String inner) {
        boolean neg = inner.startsWith("^");
        if (neg) inner = inner.substring(1);
        List<Byte> chars = new ArrayList<>();
        for (int jj = 0; jj < inner.length(); jj++) {
            byte ch;
            if (inner.charAt(jj) == '\\' && jj + 1 < inner.length()) {
                if (inner.charAt(jj + 1) == 'x' && jj + 3 < inner.length()) {
                    ch = (byte) Integer.parseInt(inner.substring(jj + 2, jj + 4), 16);
                    jj += 3;
                } else {
                    ch = (byte) unescChar(inner.charAt(jj + 1));
                    jj++;
                }
            } else {
                ch = (byte) inner.charAt(jj);
            }
            if (jj + 2 < inner.length() && inner.charAt(jj + 1) == '-') {
                int endIdx = jj + 2;
                byte endCh;
                // advance jj to the LAST char of the range-end token; the for-loop's jj++
                // then lands just past it (a relative jj += N here is off-by-one and would
                // re-read the end token's final char as a spurious extra member).
                if (inner.charAt(endIdx) == '\\' && endIdx + 1 < inner.length()) {
                    if (inner.charAt(endIdx + 1) == 'x' && endIdx + 3 < inner.length()) {
                        endCh =
                                (byte)
                                        Integer.parseInt(
                                                inner.substring(endIdx + 2, endIdx + 4), 16);
                        jj = endIdx + 3;
                    } else {
                        endCh = (byte) unescChar(inner.charAt(endIdx + 1));
                        jj = endIdx + 1;
                    }
                } else {
                    endCh = (byte) inner.charAt(endIdx);
                    jj = endIdx;
                }
                for (int x = Byte.toUnsignedInt(ch); x <= Byte.toUnsignedInt(endCh); x++)
                    chars.add((byte) x);
            } else {
                chars.add(ch);
            }
        }
        return new Rule.Element.CharClass(chars, neg);
    }

    static int findMatchingBracket(String s, int start) {
        int d = 1;
        for (int j = start + 1; j < s.length(); j++) {
            char c = s.charAt(j);
            if (c == '\\') {
                j++;
                continue;
            }
            if (c == '[') d++;
            else if (c == ']' && --d == 0) return j;
        }
        return -1;
    }

    /**
     * Consumes a trailing repetition modifier at {@code i} - {@code *}, {@code +}, {@code ?}, or
     * GBNF's bounded {@code {m}} / {@code {m,}} / {@code {m,n}} - wrapping the LAST parsed element.
     * Returns the index just past the modifier ({@code i} unchanged when none).
     */
    private static int applyMod(String body, int i, List<Rule.Element> res) {
        // whitespace between an element and its modifier is insignificant in GBNF ("a" {2})
        int j = i;
        while (j < body.length() && (body.charAt(j) == ' ' || body.charAt(j) == '\t')) j++;
        if (j >= body.length() || res.isEmpty()) return i;
        int min;
        int max;
        switch (body.charAt(j)) {
            case '*' -> {
                min = 0;
                max = -1;
            }
            case '+' -> {
                min = 1;
                max = -1;
            }
            case '?' -> {
                min = 0;
                max = 1;
            }
            case '{' -> {
                int close = body.indexOf('}', j);
                if (close < 0) return i;
                String spec = body.substring(j + 1, close).trim();
                int comma = spec.indexOf(',');
                try {
                    if (comma < 0) {
                        min = Integer.parseInt(spec);
                        max = min;
                    } else {
                        min = Integer.parseInt(spec.substring(0, comma).trim());
                        String hi = spec.substring(comma + 1).trim();
                        max = hi.isEmpty() ? -1 : Integer.parseInt(hi);
                    }
                } catch (NumberFormatException notARepetition) {
                    return i; // not a repetition spec: leave the brace alone
                }
                res.add(new Rule.Element.Repetition(res.removeLast(), min, max));
                return close + 1;
            }
            default -> {
                return i;
            }
        }
        res.add(new Rule.Element.Repetition(res.removeLast(), min, max));
        return j + 1;
    }

    static int findMatchingParen(String s, int start) {
        // parens inside string literals OR char classes don't count (a class may hold a quote,
        // a literal may hold unbalanced parens)
        int d = 1, end = start + 1;
        boolean inStr = false, escape = false, inClass = false;
        while (end < s.length() && d > 0) {
            char c = s.charAt(end);
            if (inStr) {
                if (c == '\\') escape = !escape;
                else if (c == '"' && !escape) inStr = false;
                else escape = false;
            } else if (inClass) {
                if (c == '\\') escape = !escape;
                else if (c == ']' && !escape) inClass = false;
                else escape = false;
            } else if (c == '"') inStr = true;
            else if (c == '[') inClass = true;
            else if (c == '(') d++;
            else if (c == ')') d--;
            end++;
        }
        return d == 0 ? end : -1;
    }

    static String unescape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(++i);
                switch (n) {
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'x' -> {
                        if (i + 2 < s.length()) {
                            String hex = s.substring(i + 1, i + 3);
                            sb.append((char) Integer.parseInt(hex, 16));
                            i += 2;
                        } else sb.append('x');
                    }
                    default -> sb.append(n);
                }
            } else sb.append(c);
        }
        return sb.toString();
    }

    static char unescChar(char c) {
        return switch (c) {
            case 'n' -> '\n';
            case 'r' -> '\r';
            case 't' -> '\t';
            default -> c;
        };
    }

    // ---- Rule IR -----------------------------------------------------------

    record Rule(int id, List<Element> body) {
        sealed interface Element {
            record Value(byte b) implements Element {}

            record Dot() implements Element {}

            record CharClass(List<Byte> chars, boolean neg) implements Element {}

            record Ref(int ruleId) implements Element {}

            record Group(List<Element> children) implements Element {}

            record Repetition(Element child, int min, int max) implements Element {}

            record Pipe() implements Element {}
        }
    }
}
