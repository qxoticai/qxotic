package com.qxotic.jinfer.llm;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.WeakReference;
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
 * <p>A grammar is a list of rules, one {@code name ::= body} per line. <b>The rule named {@code
 * root} is the start rule</b>, wherever it is declared (llama.cpp's contract); a grammar that
 * declares no {@code root} is refused. A line without {@code ::=} continues the previous rule
 * (alternatives may sit on their own lines); {@code #} starts a comment to end of line (except
 * inside a literal or class). Rule names are Java identifiers plus hyphens ({@code kebab-case}
 * works). Bodies compose:
 *
 * <ul>
 *   <li>{@code "literal"} — its UTF-8 bytes; escapes {@code \" \\ \n \r \t}, and {@code \xNN} for
 *       the raw byte 0xNN.
 *   <li>{@code [abc]} {@code [a-z0-9]} {@code [^,\n]} — BYTE classes: members, ranges, negation,
 *       the same escapes. {@code .} matches any one byte. The engine matches UTF-8 bytes, not code
 *       points (llama.cpp matches code points): a non-ASCII character inside a class is its bytes
 *       as separate members, so put non-ASCII text in literals, and read {@code [^"]*} as "any
 *       bytes but a quote", which is what free text wants.
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
 *   <li><b>Reasoning stays free:</b> driven through the chat engine's constrained path, the grammar
 *       binds only the output channel - think spans sample unconstrained.
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
    // the decoded byte view of a whole vocabulary, computed once per Vocab: term grammars
    // compile per REQUEST (selections are composed, not cached), and re-decoding 100k+ tokens
    // per compile was the dominant cost
    private static final Map<Vocab, byte[][]> BYTE_TABLES =
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
        return WRAPPERS.computeIfAbsent(tok, TokenizerVocab::new);
    }

    /**
     * The {@link Vocab} over a tokenizer, holding it weakly: this is the VALUE of a weak-keyed map
     * whose KEY is the tokenizer, and a strong reference from value to key would pin every
     * tokenizer that ever compiled a grammar (and its byte table and mask caches) for the JVM's
     * lifetime. A Vocab in use belongs to a live model, which holds its tokenizer strongly.
     */
    private static final class TokenizerVocab implements Vocab {
        private final WeakReference<Tokenizer> tokenizer;

        TokenizerVocab(Tokenizer tokenizer) {
            this.tokenizer = new WeakReference<>(tokenizer);
        }

        private Tokenizer tokenizer() {
            Tokenizer t = tokenizer.get();
            if (t == null) throw new IllegalStateException("the tokenizer of this grammar is gone");
            return t;
        }

        @Override
        public int size() {
            return tokenizer().vocabulary().size();
        }

        @Override
        public byte[] bytes(int id) {
            // Specials are CONTROL, not content, whatever their literal rendering (LFM2's
            // <|im_end|> decodes to its 10-char string). Empty bytes makes every special
            // unsamplable mid-grammar and samplable exactly at accept states - the model can
            // always END a completed constrained span with its natural stop token.
            Tokenizer t = tokenizer();
            if (SpecialTokens.isSpecial(t, id)) return NO_BYTES;
            return t.decodeBytes(new int[] {id});
        }
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

    // ---- programmatic grammars (token-identity terminals) ------------------

    /**
     * A programmatic grammar term - the composition layer for grammars that interleave plain bytes
     * with CONTROL TOKENS matched by identity, which no byte-level GBNF can express: a special
     * contributes empty bytes (so content can never mint it), and for exactly the same reason no
     * byte literal can ever match it. {@link Token} is the one terminal that names a token id
     * directly.
     *
     * <p>Deliberately NOT a GBNF syntax extension: GBNF text stays the byte-only front door
     * (llama.cpp-compatible, safe to accept from requests), while identity terminals exist only
     * here - reachable from trusted composing code, unwritable in a user-supplied grammar.
     */
    public sealed interface Term {
        /** Plain bytes (UTF-8), matched byte-by-byte. */
        record Text(String text) implements Term {
            public Text {
                if (text == null) throw new IllegalArgumentException("null text");
            }
        }

        /**
         * ONE vocabulary token, matched by ID - whatever its byte view (a mistyped special
         * included: the id is the term author's assertion).
         */
        record Token(int id) implements Term {
            public Token {
                if (id < 0) throw new IllegalArgumentException("negative token id");
            }
        }

        /** A byte-level GBNF fragment embedded whole; its root rule becomes this term. */
        record Gbnf(String source) implements Term {
            public Gbnf {
                if (source == null || source.isBlank())
                    throw new IllegalArgumentException("empty GBNF fragment");
            }
        }

        record Seq(List<Term> parts) implements Term {
            public Seq {
                parts = List.copyOf(parts);
            }
        }

        record Alt(List<Term> options) implements Term {
            public Alt {
                options = List.copyOf(options);
                if (options.isEmpty()) throw new IllegalArgumentException("empty alternation");
            }
        }

        /** {@code max} -1 = unbounded. */
        record Rep(Term child, int min, int max) implements Term {
            public Rep {
                if (min < 0 || max < -1 || (max >= 0 && max < min))
                    throw new IllegalArgumentException("bad repetition {" + min + "," + max + "}");
            }
        }

        static Term text(String text) {
            return new Text(text);
        }

        static Term token(int id) {
            return new Token(id);
        }

        static Term gbnf(String source) {
            return new Gbnf(source);
        }

        static Term seq(Term... parts) {
            return new Seq(List.of(parts));
        }

        static Term alt(Term... options) {
            return new Alt(List.of(options));
        }

        static Term rep(Term child, int min, int max) {
            return new Rep(child, min, max);
        }
    }

    public static Spec of(Term root, Tokenizer t) {
        return of(root, vocab(t));
    }

    /**
     * Compiles a term over the vocabulary. Uncached on purpose: term grammars are composed per
     * request from parts the caller already caches (a GBNF payload inside one still hits the
     * fragment-independent parse each compile, which is cheap; the per-state mask cache, where the
     * real cost lives, is per-Spec as always).
     */
    public static Spec of(Term root, Vocab v) {
        List<Rule> rules = new ArrayList<>();
        rules.add(null); // reserve the root id; lowering appends embedded fragments' rules
        List<Rule.Element> body = new ArrayList<>();
        lower(root, body, rules);
        rules.set(0, new Rule(0, body));
        int vs = v.size();
        requireIds(body, vs);
        for (Rule r : rules) if (r != null) requireIds(r.body(), vs);
        return new Spec(CFG.compile(rules), byteTable(v));
    }

    /** An out-of-vocabulary identity terminal would compile to a silently DEAD grammar. */
    private static void requireIds(List<Rule.Element> body, int vocab) {
        for (Rule.Element e : body) {
            switch (e) {
                case Rule.Element.TokenId(int id) -> {
                    if (id >= vocab)
                        throw new IllegalArgumentException(
                                "token id " + id + " is outside this vocabulary (" + vocab + ")");
                }
                case Rule.Element.Group(List<Rule.Element> kids) -> requireIds(kids, vocab);
                case Rule.Element.Repetition(Rule.Element child, int m, int x) ->
                        requireIds(List.of(child), vocab);
                default -> {}
            }
        }
    }

    private static void lower(Term t, List<Rule.Element> body, List<Rule> rules) {
        switch (t) {
            case Term.Text(String s) -> {
                for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
                    body.add(new Rule.Element.Value(b));
                }
            }
            case Term.Token(int id) -> body.add(new Rule.Element.TokenId(id));
            case Term.Seq(List<Term> parts) -> {
                for (Term p : parts) lower(p, body, rules);
            }
            case Term.Alt(List<Term> options) -> {
                List<Rule.Element> group = new ArrayList<>();
                for (int i = 0; i < options.size(); i++) {
                    if (i > 0) group.add(new Rule.Element.Pipe());
                    lower(options.get(i), group, rules);
                }
                body.add(new Rule.Element.Group(group));
            }
            case Term.Rep(Term child, int min, int max) -> {
                List<Rule.Element> kid = new ArrayList<>();
                lower(child, kid, rules);
                Rule.Element unit = kid.size() == 1 ? kid.get(0) : new Rule.Element.Group(kid);
                body.add(new Rule.Element.Repetition(unit, min, max));
            }
            case Term.Gbnf(String source) -> {
                List<Rule> fragment = parse(source);
                if (fragment.isEmpty()) {
                    throw new IllegalArgumentException("unparseable GBNF fragment: " + source);
                }
                // embed whole with rule ids shifted past everything appended so far; the
                // fragment's root (its id 0) is referenced here, names never collide (ids only)
                int base = rules.size();
                for (Rule r : fragment) {
                    rules.add(new Rule(r.id() + base, shift(r.body(), base)));
                }
                body.add(new Rule.Element.Ref(base));
            }
        }
    }

    private static List<Rule.Element> shift(List<Rule.Element> body, int base) {
        List<Rule.Element> out = new ArrayList<>(body.size());
        for (Rule.Element e : body) out.add(shift(e, base));
        return out;
    }

    private static Rule.Element shift(Rule.Element e, int base) {
        return switch (e) {
            case Rule.Element.Ref(int rid) -> new Rule.Element.Ref(rid + base);
            case Rule.Element.Group(List<Rule.Element> kids) ->
                    new Rule.Element.Group(shift(kids, base));
            case Rule.Element.Repetition(Rule.Element child, int min, int max) ->
                    new Rule.Element.Repetition(shift(child, base), min, max);
            default -> e; // Value, Dot, CharClass, Pipe, TokenId carry no rule ids
        };
    }

    static Spec build(String gbnf, Vocab v) {
        if (gbnf.isBlank()) return Spec.DISABLED; // the absence of a grammar, not a broken one
        List<Rule> rules = parse(gbnf);
        if (rules.isEmpty()) {
            // text with no "::=" rule is a syntax error: the pass-through spec would silently
            // generate unconstrained under a caller that believes otherwise
            throw new IllegalArgumentException("unparseable GBNF: no 'name ::= ...' rule found");
        }
        return new Spec(CFG.compile(rules), byteTable(v));
    }

    private static byte[][] byteTable(Vocab v) {
        return BYTE_TABLES.computeIfAbsent(
                v,
                k -> {
                    byte[][] table = new byte[k.size()][];
                    for (int t = 0; t < table.length; t++) table[t] = k.bytes(t);
                    return table;
                });
    }

    // ---- Compiled CFG (byte-level pushdown grammar) ------------------------
    //
    // The grammar is flattened into "slots". A slot is one of:
    //   TERM  — a 256-bit byte set + a continuation slot (the next slot after a matching byte)
    //   TOKEN - ONE vocabulary token id + a continuation slot, matched by IDENTITY (zero bytes)
    //   REF   — a rule id + a return slot (where to continue once that rule completes)
    //   END   — marks the end of a rule alternative (pop a frame)
    // A rule is a set of alternative entry slots. Groups and repetitions are desugared into
    // anonymous rules so every leaf is a TERM/TOKEN/REF/END - no nesting survives into the
    // matcher. TOKEN slots come only from programmatic terms (never GBNF text, which stays
    // byte-only): they are what lets a trusted grammar say "here comes <|constrain|>" while
    // content grammars still cannot name a control token at all.

    static final byte T_TERM = 0, T_REF = 1, T_END = 2, T_TOKEN = 3;

    static final class CFG {
        final byte[] kind; // T_TERM | T_TOKEN | T_REF | T_END
        final int[] data; // TERM: terminal index;  TOKEN: token id;  REF: rule id;  END: unused
        final int[] next; // TERM/TOKEN/REF: continuation slot (>=0);  END: unused
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
                    case Rule.Element.TokenId(int id) -> slot(T_TOKEN, id, cont);
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
     * <p>Most callers want the assembled path instead: the chat engine's constrained generation
     * wires this into the sampling stack (think spans free, output bound, dead-end stop).
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
            // Fast path (the dominant decode case): every stack already has a TERM or TOKEN on
            // top, so the closure is just dedup - no worklist, no seen-set, no StackKey
            // allocations.
            boolean anyEps = false;
            for (int[] s : raws) {
                if (s.length == 0) {
                    anyEps = true;
                    break;
                }
                byte k = cfg.kind[s[s.length - 1]];
                if (k != T_TERM && k != T_TOKEN) {
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
                    case T_TERM, T_TOKEN -> {
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
                // a TOKEN top never matches a byte: identity terminals cross in advance() only
                if (cfg.kind[top] == T_TERM && cfg.termHas(cfg.data[top], b)) {
                    raws.add(replaceLast(s, cfg.next[top]));
                }
            }
            return raws;
        }

        /**
         * Walk {@code len} bytes from a ready set; returns the surviving stacks RAW (the last step
         * unexpanded, so a caller can union them with identity crossings before one final closure),
         * or null if the bytes cannot be consumed (the grammar rejects them).
         */
        List<int[]> walkRaw(List<int[]> ready, byte[] bytes, int len) {
            List<int[]> cur = ready;
            List<int[]> raws = null;
            for (int i = 0; i < len; i++) {
                raws = step(cur, bytes[i] & 0xFF);
                if (raws.isEmpty()) return null;
                if (i + 1 < len) cur = expandSet(raws).ready();
            }
            return raws;
        }

        /**
         * One TOKEN against the state: the union of identity crossings (stacks whose top names
         * exactly this id, whatever the token's byte view - a pinned mistyped special included) and
         * the byte walk of the token's decoded bytes. Returns the resulting state, a DEAD state
         * (empty, non-accepting) when the token fits neither way, or null for the legacy no-op
         * case: an empty-byte token (a special) crossing no identity slot never advances and never
         * kills the walk - EOS at a dead end, a stop sampled at an accept state.
         */
        State advance(List<int[]> ready, int token) {
            List<int[]> raws = new ArrayList<>();
            for (int[] s : ready) {
                int top = s[s.length - 1];
                if (cfg.kind[top] == T_TOKEN && cfg.data[top] == token) {
                    raws.add(replaceLast(s, cfg.next[top]));
                }
            }
            byte[] bs = tokenBytes[token];
            if (bs.length == 0 && raws.isEmpty()) return null;
            if (bs.length > 0) {
                List<int[]> viaBytes = walkRaw(ready, bs, bs.length);
                if (viaBytes != null) raws.addAll(viaBytes);
            }
            return expandSet(raws);
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
            // First-byte filter: the union of all ready byte terminals. A token whose first byte
            // is not in it cannot match - reject in O(1) without a full walk (most of the vocab,
            // in practice). TOKEN tops admit their id by IDENTITY instead: pre-marked directly,
            // whatever that token's byte view.
            long[] firsts = new long[4];
            for (int[] s : ready) {
                int top = s[s.length - 1];
                if (cfg.kind[top] == T_TOKEN) {
                    int id = cfg.data[top]; // in-vocab by construction: of(Term,..) validates
                    m[id >> 6] |= 1L << (id & 63);
                    continue;
                }
                long[] tm = cfg.terms[cfg.data[top]];
                for (int i = 0; i < 4; i++) firsts[i] |= tm[i];
            }
            for (int t = 0; t < vocab; t++) {
                if ((m[t >> 6] & (1L << (t & 63))) != 0) continue; // admitted by identity already
                byte[] bs = tokenBytes[t];
                boolean ok;
                if (bs.length == 0) ok = accepting;
                else {
                    int f = bs[0] & 0xFF;
                    ok =
                            (firsts[f >>> 6] & (1L << (f & 63))) != 0
                                    && walkRaw(ready, bs, bs.length) != null;
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
         * A prefix grammar reaching this state has done its job; a prefix-pinning sampler then
         * releases the mask (a dead state - no match, not accepting - is not exhausted).
         */
        public boolean exhausted() {
            return spec.cfg != null && accepting && ready.isEmpty();
        }

        /**
         * True when the grammar is fully matched HERE, whether or not continuations remain - the
         * "may stop now" reading a close-less region needs ({@code exhausted} is the stricter
         * "nothing left at all"). A schema grammar with a whitespace-tolerant tail accepts after
         * the closing brace while whitespace continuations stay ready.
         */
        public boolean accepting() {
            return spec.cfg != null && accepting;
        }

        /**
         * The admissible-token mask for the CURRENT state as one bit per id ({@code long[]}, bit
         * {@code t} = token {@code t} admissible) - a fresh copy of the cached mask, for callers
         * that UNION admission across parallel cursors before masking once. A DISABLED spec returns
         * null (everything admissible).
         */
        public long[] admissible() {
            if (spec.cfg == null) return null;
            return spec.maskFor(ready, accepting).clone();
        }

        /**
         * Masks {@code logits} to grammar-allowed tokens; returns whether any token remains. A
         * DISABLED spec is a pass-through (no masking, always true).
         */
        public boolean maskLogits(MemoryView<?> logits) {
            if (spec.cfg == null) return true;
            int vocab = spec.tokenBytes.length;
            if (vocab == 0) return false;
            Views.requireF32(logits, "logits");
            Views.requireContiguous(logits, "logits");
            if (logits.shape().size() < vocab) {
                throw new IllegalArgumentException(
                        "logits: " + logits.shape().size() + " elements for vocabulary " + vocab);
            }
            MemoryView<MemorySegment> memory = Views.castToSegmentBacked(logits, "logits");
            Views.checkAlive(memory, "logits");
            long[] mask = spec.maskFor(ready, accepting);
            boolean any = false;
            for (int w = 0; w < mask.length; w++) {
                long bits = mask[w];
                if (bits == 0) { // 64 rejected: write without per-bit tests
                    int base = w << 6;
                    for (int b = 0; b < 64 && base + b < vocab; b++)
                        setNegativeInfinity(memory, base + b);
                    continue;
                }
                any = true; // this word admits something
                if (bits == -1L) continue; // 64 allowed: nothing to mask
                int base = w << 6;
                for (int b = 0; b < 64 && base + b < vocab; b++) {
                    if ((bits & (1L << b)) == 0) setNegativeInfinity(memory, base + b);
                }
            }
            return any;
        }

        private static void setNegativeInfinity(
                MemoryView<MemorySegment> logits, long elementOffset) {
            logits.memory()
                    .base()
                    .set(
                            ValueLayout.JAVA_FLOAT,
                            logits.byteOffset() + elementOffset * Float.BYTES,
                            Float.NEGATIVE_INFINITY);
        }

        /**
         * Consume a chosen token, advancing the grammar: identity slots naming exactly this id
         * cross, and the token's decoded bytes walk the byte terminals - both interpretations
         * survive when both fit. An empty-byte token (EOS/control) crossing no identity slot does
         * not advance; an impossible token drives the cursor to a dead state.
         */
        public void advanceWith(int token) {
            tryAdvance(token);
        }

        /**
         * As {@link #advanceWith}, reporting whether the token was actually CONSUMED and the walk
         * survives: false for the unexpected-special no-op (state unchanged) and for a token that
         * drives the cursor dead - a caller enforcing a reply language treats both as "this token
         * does not belong here". A DISABLED spec consumes everything.
         */
        public boolean tryAdvance(int token) {
            if (spec.cfg == null) return true;
            if (token < 0 || token >= spec.tokenBytes.length) return false;
            State st = spec.advance(ready, token);
            if (st == null) return false;
            ready = st.ready();
            accepting = st.accept();
            return !ready.isEmpty() || accepting;
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
     * const}, {@code anyOf}/{@code oneOf}, and {@code $ref} into the root document ({@code $defs},
     * {@code definitions}, any local JSON Pointer) — how a generated schema spells a type it uses
     * twice, and the only way a RECURSIVE type can be written at all. Object properties are emitted
     * in the order of {@code required} (or, when {@code required} is absent, all declared
     * properties); other keywords ({@code patternProperties}, numeric/length bounds, …) are ignored
     * — the result is always valid JSON satisfying the supported constraints, never a broken
     * grammar.
     */
    static Spec fromSchema(Map<String, Object> schema, Vocab v) {
        return of(Schema.toGbnf(schema, true), v);
    }

    /**
     * A grammar admitting exactly the JSON documents valid under {@code schema} (parsed JSON Schema
     * map): types, required/optional properties, enums, nesting - no other keys, no other shapes.
     * Whitespace bounded so a reluctant model cannot stall on it.
     */
    public static Spec fromSchema(Map<String, Object> schema, Tokenizer t) {
        return fromSchema(schema, vocab(t));
    }

    /**
     * The schema's GBNF source - for embedding a schema payload inside a larger composed grammar
     * ({@code Term.Gbnf}, reply-language argument regions). {@link #fromSchema} is this compiled.
     */
    public static String schemaGbnf(Map<String, Object> schema) {
        return Schema.toGbnf(schema, true);
    }

    /** The schemaless-JSON source ({@link #json}'s grammar) - the JSON-mode content hole. */
    public static String jsonGbnf() {
        return JSON_GRAMMAR;
    }

    /**
     * {@link #schemaGbnf} without LEADING whitespace - the reply-language content-hole form. At a
     * dispatch point interstitial newlines are scaffold between spans, and a hole whose entry set
     * admits them swallows the model into content it meant as spacing before a tool call (observed
     * on LFM2.5: "I should call" in the reasoning, then a hallucinated schema answer).
     */
    public static String schemaHoleGbnf(Map<String, Object> schema) {
        return Schema.toGbnf(schema, false);
    }

    /** Translates a JSON Schema node tree into a GBNF grammar string. */
    static final class Schema {
        private final StringBuilder rules = new StringBuilder();
        private int counter;
        // the document $ref pointers resolve against, and one rule per pointer already emitted
        private Map<String, Object> root = Map.of();
        private final Map<String, String> refRules = new LinkedHashMap<>();
        private final Set<String> formats = new HashSet<>(); // format rules already emitted

        static String toGbnf(Map<String, Object> schema, boolean leadingWs) {
            Schema s = new Schema();
            s.root = schema;
            // shared leaf rules (any-JSON fallbacks + scalars)
            // BOUNDED whitespace (llama.cpp-style): unbounded ws lets a reluctant model stall
            // forever without progress, growing a fresh matcher state (and a full-vocab mask
            // recompute) per whitespace token. Eight chars covers pretty-printing.
            s.rules.append("ws ::= [ \\t\\n\\r]{0,8}\n");
            // one JSON character, named so minLength/maxLength can bound a repetition of it
            s.rules.append(
                    "char ::= [^\"\\\\\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\""
                            + " [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])\n");
            s.rules.append("string ::= \"\\\"\" char* \"\\\"\"\n");
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
            return "root ::= " + (leadingWs ? "ws (" : "(") + root + ") ws\n" + s.rules;
        }

        /**
         * A {@code $ref} as a NAMED rule, so the pointed-at shape is written once however many
         * times it is referenced. The name is registered BEFORE the target's body is built: a
         * definition that reaches itself (a tree node, a linked list, mutually recursive types)
         * then closes into a recursive GBNF rule instead of recursing forever here. A pointer that
         * resolves to nothing - an external document, a name the producer never emitted - degrades
         * to {@code value}: this compiler's law is a permissive grammar, never a broken one.
         */
        private String refRule(String pointer) {
            String named = refRules.get(pointer);
            if (named != null) return named;
            Object target = resolve(pointer);
            if (target == null) return "value";
            String name = "r" + (counter++);
            refRules.put(pointer, name);
            String b = body(target); // built BEFORE appending: nested rules own whole lines
            rules.append(name).append(" ::= ").append(b).append("\n");
            return name;
        }

        /**
         * RFC 6901 JSON Pointer against the root schema ({@code #}, {@code #/$defs/Name}, {@code
         * #/definitions/Name/properties/x}, array indices included); {@code null} when the pointer
         * leaves the document or names something absent.
         */
        private Object resolve(String pointer) {
            if (!pointer.startsWith("#")) return null; // another document: nothing to resolve with
            Object node = root;
            for (String segment : pointer.substring(1).split("/")) {
                if (segment.isEmpty()) continue; // the empty head of "#/..."
                String key = segment.replace("~1", "/").replace("~0", "~"); // ~1 first (RFC 6901)
                node =
                        switch (node) {
                            case Map<?, ?> map -> map.get(key);
                            case List<?> list -> element(list, key);
                            default -> null;
                        };
                if (node == null) return null;
            }
            return node;
        }

        /** A pointer segment indexing an array: the element, or null when it is not an index. */
        private static Object element(List<?> list, String key) {
            try {
                int i = Integer.parseInt(key);
                return i >= 0 && i < list.size() ? list.get(i) : null;
            } catch (NumberFormatException notAnIndex) {
                return null;
            }
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
            // $ref REPLACES the node it sits on (draft semantics), so it answers before anything
            // else this node might also carry
            if (m.get("$ref") instanceof String pointer) return refRule(pointer);
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
            // "format" shapes a STRING, and a node may carry it without saying so (the schema
            // {"format":"date"} is a date string). An unknown format degrades to a plain string:
            // the compiler's law is permissive, never broken.
            if ((m.get("type") == null || "string".equals(m.get("type")))
                    && m.get("format") instanceof String fmt) {
                String rule = formatRule(fmt);
                if (rule != null) return rule;
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
            // "type" is not mandatory in JSON Schema, and generators leave it out: a node carrying
            // "properties" IS an object, one carrying "items" IS an array. Without this the whole
            // node fell through to any-JSON and constrained nothing at all - the shape langchain4j
            // sends for a nested record, silently unconstrained. llama.cpp infers the same way.
            if (m.containsKey("properties") || m.containsKey("required")) return objectBody(m);
            if (m.containsKey("items")) return arrayBody(m);
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
                case "string" -> stringBody(m);
                default -> "value";
            };
        }

        /**
         * JSON Schema's object rule, as llama.cpp builds it: the required properties in the
         * schema's DECLARATION order, every one present, then the optional properties as an ORDERED
         * SUBSET, also in declaration order (any of them may be omitted, none may be invented).
         * Property order is fixed by the grammar, a documented limitation shared with llama.cpp.
         */
        @SuppressWarnings("unchecked")
        private String objectBody(Map<String, Object> m) {
            Object propsObj = m.get("properties");
            if (!(propsObj instanceof Map) || ((Map<?, ?>) propsObj).isEmpty())
                return "\"{\" ws \"}\"";
            Map<String, Object> props = (Map<String, Object>) propsObj;
            Set<String> required = new HashSet<>();
            if (m.get("required") instanceof List<?> req)
                for (Object k : req) required.add(String.valueOf(k));
            // BOTH groups walk the DECLARATION order, and only membership decides which group a
            // property joins: the "required" array states WHICH properties are mandatory, never
            // the order they appear in. Ordering by that array instead (this compiler's first
            // reading) put {"a":..,"b":..} in the language of a schema declaring b before a - a
            // shape llama.cpp rejects, and the wrong bet besides: a generated schema lists
            // properties in its source type's field order, which is the order a model reading
            // that schema emits them.
            List<String> head = new ArrayList<>(), tail = new ArrayList<>();
            for (String k : props.keySet())
                (required.contains(k) ? head : tail).add(pair(k, props.get(k)));
            String optional = tail.isEmpty() ? null : optionalSubset(tail, 0);
            if (head.isEmpty()) {
                return optional == null
                        ? "\"{\" ws \"}\""
                        : "\"{\" ws (" + optional + ")? ws \"}\"";
            }
            String pairs = String.join(" ws \",\" ws ", head);
            return optional == null
                    ? "\"{\" ws " + pairs + " ws \"}\""
                    : "\"{\" ws " + pairs + " (ws \",\" ws " + optional + ")? ws \"}\"";
        }

        /** {@code "key" ws ":" ws <value rule>}. */
        private String pair(String key, Object schema) {
            return gbnfLiteral("\"" + jsonEsc(key) + "\"") + " ws \":\" ws " + rule(schema);
        }

        /**
         * The non-empty ordered subsets of {@code pairs[from..)}, one named rule per suffix: {@code
         * S_i ::= kv_i (ws "," ws S_i+1)? | S_i+1}, {@code S_n ::= kv_n}. Returns the name of
         * {@code S_from}.
         */
        private String optionalSubset(List<String> pairs, int from) {
            String kv = pairs.get(from);
            String body =
                    from + 1 == pairs.size()
                            ? kv
                            : kv
                                    + " (ws \",\" ws "
                                    + optionalSubset(pairs, from + 1)
                                    + ")? | "
                                    + optionalSubset(pairs, from + 1);
            String name = "r" + (counter++);
            rules.append(name).append(" ::= ").append(body).append("\n");
            return name;
        }

        /**
         * The {@code format} shapes llama.cpp defines, emitted on first use so an ordinary grammar
         * does not carry them. STRUCTURAL only: the month is 01-12 and the day 01-31, so February
         * 31st parses - a grammar can pin a shape, never a calendar.
         */
        private String formatRule(String format) {
            String body =
                    switch (format) {
                        case "date" ->
                                "[0-9]{4} \"-\" (\"0\" [1-9] | \"1\" [0-2]) \"-\" (\"0\" [1-9]"
                                        + " | [1-2] [0-9] | \"3\" [0-1])";
                        case "time" ->
                                "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9]"
                                        + " (\".\" [0-9]{3})? (\"Z\" | (\"+\" | \"-\") ([01] [0-9]"
                                        + " | \"2\" [0-3]) \":\" [0-5] [0-9])";
                        case "date-time" -> "fmt-date \"T\" fmt-time";
                        case "uuid" ->
                                "[0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\""
                                        + " [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12}";
                        default -> null;
                    };
            if (body == null) return null;
            if ("date-time".equals(format)) { // its body names the other two
                formatRule("date");
                formatRule("time");
            }
            if (formats.add(format))
                rules.append("fmt-").append(format).append(" ::= ").append(body).append("\n");
            return "\"\\\"\" fmt-" + format + " \"\\\"\"";
        }

        /** {@code minLength}/{@code maxLength} bound a repetition of {@code char}. */
        private String stringBody(Map<String, Object> m) {
            long min = bound(m, "minLength", 0), max = bound(m, "maxLength", -1);
            if (min == 0 && max < 0) return "string";
            return "\"\\\"\" " + repeat("char", min, max) + " \"\\\"\"";
        }

        private String arrayBody(Map<String, Object> m) {
            // "items" as a LIST is tuple validation: one schema per position, length fixed by the
            // list. Length keywords do not apply - the tuple already states the length.
            if (m.get("items") instanceof List<?> tuple) {
                if (tuple.isEmpty()) return "\"[\" ws \"]\"";
                StringBuilder sb = new StringBuilder("\"[\" ws ");
                for (int i = 0; i < tuple.size(); i++) {
                    if (i > 0) sb.append(" ws \",\" ws ");
                    sb.append(rule(tuple.get(i)));
                }
                return sb.append(" ws \"]\"").toString();
            }
            String item = m.containsKey("items") ? rule(m.get("items")) : "value";
            long min = bound(m, "minItems", 0), max = bound(m, "maxItems", -1);
            if (min == 0 && max < 0)
                return "\"[\" ws (" + item + " (ws \",\" ws " + item + ")*)? ws \"]\"";
            // the FIRST item carries no separator, so the bounds move to the comma-led tail
            String tail = "(ws \",\" ws " + item + ")";
            if (min >= 1)
                return "\"[\" ws " + item + " " + repeat(tail, min - 1, max - 1) + " ws \"]\"";
            // nothing is required: the whole list, tail included, becomes optional
            return "\"[\" ws (" + item + " " + repeat(tail, 0, max - 1) + ")? ws \"]\"";
        }

        /**
         * {@code term{min,max}} in GBNF; {@code max < 0} is unbounded, and {0,} is just {@code *}.
         */
        private static String repeat(String term, long min, long max) {
            if (max < 0) return min == 0 ? term + "*" : term + "{" + min + ",}";
            if (max == 0) return "";
            return min == max ? term + "{" + min + "}" : term + "{" + min + "," + max + "}";
        }

        /** A non-negative integer keyword, or {@code missing} when absent or not a number. */
        private static long bound(Map<String, Object> m, String key, long missing) {
            return m.get(key) instanceof Number n && n.longValue() >= 0 ? n.longValue() : missing;
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
        // A schema that arrived as TEXT carries the parser's null SENTINEL, not a Java null (a raw
        // JsonRawSchema, Spring's outputSchema string, any Json.parse'd document). Without this the
        // stringify fallback below quoted it, so {"enum":[...,null]} could never emit null and
        // accepted the four-character string "null" instead. Caught by llama.cpp's corpus.
        if (v == null || v == Json.NULL) return "null";
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
                for (byte b : literalBytes(body.substring(i + 1, end)))
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
                    // skipping the '(' and carrying on made `root ::= (((` compile to a rule
                    // matching ONLY the empty string - so a typo produced a model that could say
                    // nothing at all, with no error. Its neighbours (unterminated string,
                    // undefined reference) already throw; this is the same kind of mistake.
                    throw new IllegalArgumentException("unbalanced '(' in rule body: " + body);
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

    /**
     * A literal's bytes: text is UTF-8, {@code \n \r \t \" \\} are their bytes, and {@code \xNN} is
     * the raw byte 0xNN - the same byte a class member {@code [\xNN]} matches, so the two spellings
     * of a non-ASCII byte sequence ({@code "\xC3\xA9"} and {@code "é"}) are one language.
     */
    static byte[] literalBytes(String s) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(s.length());
        StringBuilder text = new StringBuilder(); // pending plain text, encoded as one run
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '\\' || i + 1 >= s.length()) {
                text.append(c);
                continue;
            }
            char n = s.charAt(++i);
            if (n == 'x') {
                if (i + 2 >= s.length()) {
                    throw new IllegalArgumentException("\\x needs two hex digits: " + s);
                }
                out.writeBytes(text.toString().getBytes(StandardCharsets.UTF_8));
                text.setLength(0);
                out.write(Integer.parseInt(s.substring(i + 1, i + 3), 16));
                i += 2;
            } else {
                text.append(unescChar(n));
            }
        }
        out.writeBytes(text.toString().getBytes(StandardCharsets.UTF_8));
        return out.toByteArray();
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

            /** One vocabulary token by IDENTITY - programmatic-terms only, never parsed GBNF. */
            record TokenId(int id) implements Element {}

            record Dot() implements Element {}

            record CharClass(List<Byte> chars, boolean neg) implements Element {}

            record Ref(int ruleId) implements Element {}

            record Group(List<Element> children) implements Element {}

            record Repetition(Element child, int min, int max) implements Element {}

            record Pipe() implements Element {}
        }
    }
}
