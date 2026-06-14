package com.llama4j;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.WeakHashMap;

public final class Grammar {

    static final boolean ENABLED = !"false".equals(System.getProperty("llama.grammar"));
    static final int MAX_DFA_STATES = 2048;

    private static final Map<LFMTokenizer, Vocab> WRAPPERS = Collections.synchronizedMap(new WeakHashMap<>());
    private static final Map<Vocab, Map<String, Spec>> CACHES = Collections.synchronizedMap(new WeakHashMap<>());

    private Grammar() {}

    public interface Vocab { int size(); byte[] bytes(int tokenId); }

    static Vocab vocab(LFMTokenizer tok) {
        return WRAPPERS.computeIfAbsent(tok, t -> new Vocab() {
            public int size() { return t.vocabularySize(); }
            public byte[] bytes(int id) { return t.decodeTokenBytes(id); }
        });
    }

    private static Map<String, Spec> cache(Vocab v) {
        return CACHES.computeIfAbsent(v, k -> {
            @SuppressWarnings("serial")
            var m = new LinkedHashMap<String, Spec>(16, 0.75f, true) {
                @Override protected boolean removeEldestEntry(Map.Entry<String, Spec> e) { return size() > 32; }
            };
            return Collections.synchronizedMap(m);
        });
    }

    public static Spec json(LFMTokenizer t) { return json(vocab(t)); }
    public static Spec json(Vocab v) {
        if (!ENABLED) return Spec.DISABLED;
        return cache(v).computeIfAbsent("__json__", k -> JsonDFA.build(v));
    }

    public static Spec of(String g, LFMTokenizer t) { return of(g, vocab(t)); }
    public static Spec of(String g, Vocab v) {
        if (!ENABLED) return Spec.DISABLED;
        return cache(v).computeIfAbsent(g, k -> Spec.compile(k, v));
    }

    static final class JsonDFA {
        static Spec build(Vocab v) {
            int vs = v.size();
            byte[][] allBytes = new byte[vs][];
            byte[] firsts = new byte[vs];
            for (int t = 0; t < vs; t++) { allBytes[t] = v.bytes(t); firsts[t] = allBytes[t].length > 0 ? allBytes[t][0] : 0; }
            int states = 10;
            int[] trans = new int[states * 256];
            Arrays.fill(trans, -1);

            byte[] digits = "0123456789".getBytes(java.nio.charset.StandardCharsets.UTF_8);

            for (int w : WS_BYTES) trans[9 * 256 + w] = 9;
            trans[9 * 256 + '{'] = 1; trans[9 * 256 + '['] = 4; trans[9 * 256 + '"'] = 5;
            trans[9 * 256 + 't'] = 8; trans[9 * 256 + 'f'] = 8; trans[9 * 256 + 'n'] = 8; trans[9 * 256 + '-'] = 8;
            for (byte db : digits) trans[9 * 256 + (db & 0xFF)] = 8;

            System.arraycopy(trans, 9 * 256, trans, 0, 256);

            for (int w : WS_BYTES) trans[1 * 256 + w] = 1;
            trans[1 * 256 + '}'] = 0; trans[1 * 256 + '"'] = 5;
            System.arraycopy(trans, 1 * 256, trans, 2 * 256, 256);
            System.arraycopy(trans, 0, trans, 3 * 256, 256);

            for (int w : WS_BYTES) trans[4 * 256 + w] = 4;
            System.arraycopy(trans, 9 * 256, trans, 4 * 256, 256);
            trans[4 * 256 + ']'] = 0;

            for (int b = 0; b < 256; b++) if (b != '"' && b != '\\') trans[5 * 256 + b] = 5;
            trans[5 * 256 + '\\'] = 6; trans[5 * 256 + '"'] = 0;

            for (int b = 0; b < 256; b++) trans[6 * 256 + b] = 5;
            trans[6 * 256 + 'u'] = 7;

            byte[] hex = "0123456789abcdefABCDEF".getBytes(java.nio.charset.StandardCharsets.UTF_8);
            for (byte hb : hex) trans[7 * 256 + (hb & 0xFF)] = 7;
            for (int b = 0; b < 256; b++) if (trans[7 * 256 + b] == -1) trans[7 * 256 + b] = 5;

            for (int b = 0; b < 256; b++) trans[8 * 256 + b] = 0;
            for (byte db : digits) trans[8 * 256 + (db & 0xFF)] = 8;
            trans[8 * 256 + 't'] = 8; trans[8 * 256 + 'r'] = 8; trans[8 * 256 + 'u'] = 8; trans[8 * 256 + 'e'] = 8;
            trans[8 * 256 + 'f'] = 8; trans[8 * 256 + 'a'] = 8; trans[8 * 256 + 'l'] = 8; trans[8 * 256 + 's'] = 8;
            trans[8 * 256 + 'n'] = 8; trans[8 * 256 + 'u'] = 8; trans[8 * 256 + 'l'] = 8;
            trans[8 * 256 + '-'] = 8; trans[8 * 256 + '.'] = 8; trans[8 * 256 + 'e'] = 8; trans[8 * 256 + 'E'] = 8; trans[8 * 256 + '+'] = 8;

            boolean[] acc = new boolean[states];
            acc[0] = true;
            return new Spec(new long[states][], allBytes, firsts, trans, acc, 9);
        }
        static final int[] WS_BYTES = {' ', '\t', '\n', '\r'};
    }

    // ---- Spec --------------------------------------------------------------

    public record Spec(long[][] stateMasks, byte[][] tokenBytes, byte[] firstBytes,
                       int[] transitions, boolean[] accepting, int startState) {
        static final Spec DISABLED = new Spec(null, null, null, null, null, -1);
        public Cursor cursor() { return new Cursor(this); }
        public boolean isValid() { return this != DISABLED && transitions != null; }

        static Spec compile(String gbnf, Vocab v) {
            int vs = v.size();
            byte[][] allBytes = new byte[vs][];
            byte[] firsts = new byte[vs];
            for (int t = 0; t < vs; t++) { allBytes[t] = v.bytes(t); firsts[t] = allBytes[t].length > 0 ? allBytes[t][0] : 0; }
            List<Rule> rules = parse(gbnf);
            FlatNFA nfa = NfaBuilder.build(rules);
            int[][] dfaRows = DfaBuilder.build(nfa);
            int ns = dfaRows.length;
            int[] trans = new int[ns * 256]; Arrays.fill(trans, -1);
            boolean[] acc = new boolean[ns];
            for (int i = 0; i < ns; i++) { acc[i] = (dfaRows[i][0] & 1) != 0; System.arraycopy(dfaRows[i], 1, trans, i * 256, 256); }
            return new Spec(new long[ns][], allBytes, firsts, trans, acc, 0);
        }
    }

    // ---- Cursor ------------------------------------------------------------

    public static final class Cursor {
        private final Spec spec;
        private final long[] scratch;
        private int state, cachedState = -1;
        Cursor(Spec spec) {
            this.spec = spec; this.state = spec.startState;
            int vs = spec.tokenBytes != null ? spec.tokenBytes.length : 0;
            this.scratch = spec.stateMasks != null && state >= 0 ? new long[(vs + 63) / 64] : null;
        }
        public void reset() { state = spec.startState; cachedState = -1; }
        public boolean maskLogits(FloatTensor logits) {
            if (spec == Spec.DISABLED) return true;
            int vocab = spec.tokenBytes.length;
            if (state < 0) { for (int i = 0; i < vocab; i++) logits.setFloat(i, Float.NEGATIVE_INFINITY); return false; }
            long[] mask = maskForState(); if (mask == null) { for (int i = 0; i < vocab; i++) logits.setFloat(i, Float.NEGATIVE_INFINITY); return false; }
            boolean any = false;
            for (int i = 0; i < vocab; i++) {
                int w = i >> 6;
                if (w < mask.length && (mask[w] & (1L << (i & 63))) != 0) any = true;
                else logits.setFloat(i, Float.NEGATIVE_INFINITY);
            }
            return any;
        }
        public void advanceWith(int token) {
            if (state < 0 || token < 0 || token >= spec.tokenBytes.length) return;
            if (spec.accepting != null && state < spec.accepting.length && spec.accepting[state]) return;
            for (byte b : spec.tokenBytes[token]) {
                if (state < 0) return;
                int off = state * 256 + (b & 0xFF);
                state = (off < 0 || off >= spec.transitions.length) ? -1 : spec.transitions[off];
            }
        }
        private long[] maskForState() {
            if (state < 0) return null;
            if (state == cachedState) return scratch;
            long[] m = spec.stateMasks[state];
            if (m != null) { System.arraycopy(m, 0, scratch, 0, Math.min(m.length, scratch.length)); cachedState = state; return m; }
            m = computeMask(); spec.stateMasks[state] = m; cachedState = state;
            System.arraycopy(m, 0, scratch, 0, Math.min(m.length, scratch.length)); return m;
        }
        private long[] computeMask() {
            int vocab = spec.tokenBytes.length;
            long[] m = new long[(vocab + 63) / 64];
            for (int t = 0; t < vocab; t++) {
                int cur = state; boolean ok = true;
                byte[] bytes = spec.tokenBytes[t];
                for (byte b : bytes) {
                    if (cur < 0) { ok = false; break; }
                    int off = cur * 256 + (b & 0xFF);
                    cur = (off < 0 || off >= spec.transitions.length) ? -1 : spec.transitions[off];
                    if (cur < 0) { ok = false; break; }
                }
                if (ok && cur >= 0) m[t >> 6] |= 1L << (t & 63);
            }
            return m;
        }
    }

    // ---- Flat NFA ----------------------------------------------------------

    static final class FlatNFA {
        static final int K_LIT = 0, K_CHARS = 1, K_EPS = 2, K_DOT = 4;
        static final int ACCEPT_MARKER = -2; // data value marking an accept EPS node
        final byte[] kind, lit;
        final int[] next, data, altData;
        final BitSet[] charSets;
        final int rule0Entry;
        FlatNFA(byte[] k, byte[] l, int[] n, int[] d, int[] a, BitSet[] c, int r) { kind = k; lit = l; next = n; data = d; altData = a; charSets = c; rule0Entry = r; }
    }

    static final class NfaBuilder {
        byte[] kind = new byte[256];
        byte[] lit = new byte[256];
        int[] next = new int[256];
        int[] data = new int[256];
        int[] altData = new int[256];
        final List<BitSet> charSets = new ArrayList<>();
        final Map<Integer, Integer> ruleEntry = new LinkedHashMap<>();
        final Map<Integer, Integer> tailCache = new java.util.HashMap<>();
        int rule0Entry, count;

        { Arrays.fill(next, -1); Arrays.fill(data, -1); Arrays.fill(altData, -1); }

        private void ensureCap(int extra) {
            int needed = count + extra;
            if (needed <= kind.length) return;
            int newLen = Math.max(kind.length * 2, needed + 64);
            kind = Arrays.copyOf(kind, newLen);
            lit = Arrays.copyOf(lit, newLen);
            next = Arrays.copyOf(next, newLen);
            data = Arrays.copyOf(data, newLen);
            altData = Arrays.copyOf(altData, newLen);
        }

        static FlatNFA build(List<Rule> rules) {
            if (rules.isEmpty()) {
                NfaBuilder b = new NfaBuilder();
                b.rule0Entry = b.addAccept(0);
                return new FlatNFA(new byte[]{FlatNFA.K_EPS}, new byte[1], new int[]{-1}, new int[]{FlatNFA.ACCEPT_MARKER}, new int[]{-1}, new BitSet[0], 0);
            }
            NfaBuilder b = new NfaBuilder();
            int[] entries = new int[rules.size()];
            for (Rule r : rules) { entries[r.id] = b.addEps(-2); b.ruleEntry.put(r.id, entries[r.id]); }
            for (Rule r : rules) { int body = b.buildAlt(r.body, r.id); b.data[entries[r.id]] = body; }
            b.rule0Entry = entries[0];
            return new FlatNFA(Arrays.copyOf(b.kind, b.count), Arrays.copyOf(b.lit, b.count),
                    Arrays.copyOf(b.next, b.count), Arrays.copyOf(b.data, b.count),
                    Arrays.copyOf(b.altData, b.count),
                    b.charSets.toArray(new BitSet[0]), b.rule0Entry);
        }

        int buildAlt(List<Rule.Element> body, int acceptRule) {
            List<List<Rule.Element>> alts = new ArrayList<>();
            List<Rule.Element> cur = new ArrayList<>();
            for (Rule.Element e : body) { if (e instanceof Rule.Element.Pipe) { if (!cur.isEmpty()) alts.add(cur); cur = new ArrayList<>(); } else cur.add(e); }
            if (!cur.isEmpty()) alts.add(cur);
            if (alts.isEmpty()) return addAccept(acceptRule);
            int acceptNode = addAccept(acceptRule);
            int result = buildSeq(alts.get(0));
            linkDirect(result, acceptNode);
            for (int i = 1; i < alts.size(); i++) {
                int n = buildSeq(alts.get(i));
                linkDirect(n, acceptNode);
                result = addEpsSplit(result, n);
            }
            tailCache.put(result, acceptNode);
            return result;
        }

        int buildSeq(List<Rule.Element> elems) {
            if (elems.isEmpty()) return -1;
            int head = buildElem(elems.get(0));
            int last = tailOf(head);
            for (int i = 1; i < elems.size(); i++) {
                int n = buildElem(elems.get(i));
                last = linkDirect(last, n);
            }
            tailCache.put(head, last);
            return head;
        }

        int buildElem(Rule.Element e) { return switch (e) {
            case Rule.Element.Value(byte b) -> addLit(b);
            case Rule.Element.Dot ignored -> {
                BitSet bs = new BitSet(); bs.set(0, 256);
                yield addCharSet(bs);
            }
            case Rule.Element.CharClass(List<Byte> chars, boolean neg) -> {
                BitSet bs = new BitSet();
                for (byte b : chars) bs.set(b & 0xFF);
                if (neg) bs.flip(0, 256);
                yield addCharSet(bs);
            }
            case Rule.Element.Ref(int rid) -> {
                Integer en = ruleEntry.get(rid);
                yield en != null ? addEps(en) : addAccept(-1);
            }
            case Rule.Element.Group(List<Rule.Element> kids) -> {
                boolean hasPipe = kids.stream().anyMatch(k -> k instanceof Rule.Element.Pipe);
                yield hasPipe ? buildAlt(kids, -1) : buildSeq(kids);
            }
            case Rule.Element.Repetition(Rule.Element child, int min, int max) -> {
                int inner = buildElem(child);
                if (min == 0 && max == 1) { // optional: E?
                    int after = addAccept(-1);
                    linkDirect(inner, after);
                    int skip = addAccept(-1);
                    yield addEpsSplit(inner, skip);
                } else if (min == 0 && max < 0) { // Kleene star: E*
                    int skip = addAccept(-1);
                    int alt = addEpsSplit(inner, skip);
                    linkDirect(inner, addEps(alt));
                    yield alt;
                } else if (min == 1 && max < 0) { // plus: E+
                    int skip = addAccept(-1);
                    int alt = addEpsSplit(inner, skip);
                    linkDirect(inner, addEps(alt));
                    yield inner;
                } else { // exact repetition: E{min}
                    int head = inner;
                    int prev = inner;
                    for (int i = 1; i < min; i++) {
                        int c = buildElem(child);
                        prev = linkDirect(prev, c);
                    }
                    yield head;
                }
            }
            case Rule.Element.Pipe ignored -> -1;
        };}

        int tailOf(int head) { return tailCache.getOrDefault(head, head); }
        int linkDirect(int prev, int nxt) {
            if (prev < 0 || nxt < 0) return prev;
            int pTail = tailOf(prev);
            NfaBuilder.this.next[pTail] = nxt;
            int nTail = tailOf(nxt);
            tailCache.put(prev, nTail);
            return nTail;
        }
        int addLit(byte b) { ensureCap(1); int n = count++; kind[n] = FlatNFA.K_LIT; lit[n] = b; return n; }
        int addCharSet(BitSet bs) { ensureCap(1); int n = count++; kind[n] = FlatNFA.K_CHARS; data[n] = charSets.size(); charSets.add(bs); return n; }
        int addAccept(int rid) { ensureCap(1); int n = count++; kind[n] = FlatNFA.K_EPS; data[n] = FlatNFA.ACCEPT_MARKER; return n; }
        int addEps(int target) { ensureCap(1); int n = count++; kind[n] = FlatNFA.K_EPS; data[n] = target; return n; }
        int addEpsSplit(int a, int b) { ensureCap(1); int n = count++; kind[n] = FlatNFA.K_EPS; data[n] = a; altData[n] = b; return n; }
    }

    // ---- DFA construction --------------------------------------------------

    static final class DfaBuilder {

        static final class BSKey {
            final long[] words;
            BSKey(BitSet bs) { words = bs.toLongArray(); }
            @Override public int hashCode() { return Arrays.hashCode(words); }
            @Override public boolean equals(Object o) {
                return o instanceof BSKey k && Arrays.equals(words, k.words);
            }
            static BSKey of(BitSet bs) { return new BSKey(bs); }
        }

        static int[][] build(FlatNFA nfa) {
            BitSet startBS = epsClos(nfa, BitSet.valueOf(new long[]{1L << nfa.rule0Entry}));
            List<BitSet> states = new ArrayList<>();
            List<int[]> rows = new ArrayList<>();
            Map<BSKey, Integer> idx = new LinkedHashMap<>();
            Queue<Integer> wl = new ArrayDeque<>();
            idx.put(BSKey.of(startBS), 0); states.add(startBS); rows.add(null); wl.add(0);

            while (!wl.isEmpty() && states.size() < MAX_DFA_STATES) {
                int si = wl.poll();
                BitSet bs = states.get(si);
                int[] row = new int[257];
                row[0] = hasAccept(nfa, bs) ? 1 : 0;
                for (int b = 0; b < 256; b++) {
                    BitSet nxt = stepAndClose(nfa, bs, (byte) b);
                    if (nxt.isEmpty()) { row[1 + b] = -1; continue; }
                    BSKey key = BSKey.of(nxt);
                    Integer nsi = idx.get(key);
                    if (nsi == null) {
                        nsi = states.size();
                        idx.put(key, nsi);
                        states.add(nxt);
                        rows.add(null);
                        wl.add(nsi);
                    }
                    row[1 + b] = nsi;
                }
                while (rows.size() <= si) rows.add(null);
                rows.set(si, row);
            }
            for (int i = 0; i < rows.size(); i++)
                if (rows.get(i) == null) { int[] r = new int[257]; Arrays.fill(r, 1, 257, -1); rows.set(i, r); }
            return rows.toArray(new int[0][]);
        }

        static BitSet stepAndClose(FlatNFA nfa, BitSet bs, byte b) {
            BitSet res = new BitSet();
            for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
                if (i >= nfa.kind.length) continue;
                if (nfa.kind[i] == FlatNFA.K_DOT) { if (nfa.next[i] >= 0) res.set(nfa.next[i]); }
                else if (nfa.kind[i] == FlatNFA.K_LIT && nfa.lit[i] == b) { if (nfa.next[i] >= 0) res.set(nfa.next[i]); }
                else if (nfa.kind[i] == FlatNFA.K_CHARS && nfa.charSets[nfa.data[i]].get(b & 0xFF)) { if (nfa.next[i] >= 0) res.set(nfa.next[i]); }
            }
            return epsClos(nfa, res);
        }

        static BitSet epsClos(FlatNFA nfa, BitSet states) {
            BitSet c = (BitSet) states.clone();
            int capacity = nfa.kind.length * 2 + 64;
            int[] q = new int[Math.min(capacity, 65536)];
            int h = 0, t = 0;
            for (int i = c.nextSetBit(0); i >= 0; i = c.nextSetBit(i + 1))
                if (t < q.length) q[t++] = i;
            while (h < t) {
                int s = q[h++];
                if (s >= nfa.kind.length) continue;
                if (nfa.kind[s] == FlatNFA.K_EPS) {
                    if (nfa.data[s] >= 0 && !c.get(nfa.data[s]) && t < q.length) { c.set(nfa.data[s]); q[t++] = nfa.data[s]; }
                    if (nfa.altData[s] >= 0 && !c.get(nfa.altData[s]) && t < q.length) { c.set(nfa.altData[s]); q[t++] = nfa.altData[s]; }
                    if (nfa.data[s] == FlatNFA.ACCEPT_MARKER && nfa.next[s] >= 0 && !c.get(nfa.next[s]) && t < q.length) { c.set(nfa.next[s]); q[t++] = nfa.next[s]; }
                }
            }
            return c;
        }

        static boolean hasAccept(FlatNFA nfa, BitSet bs) {
            for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1))
                if (i < nfa.kind.length && nfa.kind[i] == FlatNFA.K_EPS && nfa.data[i] == FlatNFA.ACCEPT_MARKER) return true;
            return false;
        }
    }

    // ---- Parser ------------------------------------------------------------

    static List<Rule> parse(String gbnf) {
        Map<String, Integer> nameToId = new LinkedHashMap<>();
        List<Rule> rules = new ArrayList<>();
        for (String raw : gbnf.split("\n")) {
            String line = stripComment(raw).trim();
            if (line.isEmpty()) continue;
            int eq = line.indexOf("::=");
            if (eq < 0) continue;
            String name = line.substring(0, eq).trim();
            if (!nameToId.containsKey(name)) { nameToId.put(name, rules.size()); rules.add(null); }
        }
        for (String raw : gbnf.split("\n")) {
            String line = stripComment(raw).trim();
            if (line.isEmpty()) continue;
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
        boolean inStr = false;
        boolean escape = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inStr) { if (c == '\\') escape = !escape; else if (c == '"' && !escape) inStr = false; else escape = false; }
            else if (c == '"') inStr = true;
            else if (c == '#') return line.substring(0, i);
        }
        return line;
    }

    private static List<Rule.Element> parseBody(String body, Map<String, Integer> rules) {
        List<Rule.Element> res = new ArrayList<>();
        int i = 0;
        while (i < body.length()) {
            char c = body.charAt(i);
            if (c == ' ' || c == '\t') { i++; continue; }
            if (c == '"') {
                int end = body.indexOf('"', i + 1);
                if (end < 0) { i++; continue; }
                String s = unescape(body.substring(i + 1, end));
                for (byte b : s.getBytes(java.nio.charset.StandardCharsets.UTF_8))
                    res.add(new Rule.Element.Value(b));
                i = end + 1;
                char mod = i < body.length() ? body.charAt(i) : 0;
                if (mod == '*' || mod == '+' || mod == '?') {
                    Rule.Element last = res.removeLast();
                    int min = mod == '+' ? 1 : 0;
                    int max = mod == '?' ? 1 : -1;
                    res.add(new Rule.Element.Repetition(last, min, max));
                    i++;
                }
            }
            else if (c == '[') {
                int end = findMatchingBracket(body, i);
                if (end < 0) { i++; continue; }
                String inner = body.substring(i + 1, end);
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
                        if (inner.charAt(endIdx) == '\\' && endIdx + 1 < inner.length()) {
                            if (inner.charAt(endIdx + 1) == 'x' && endIdx + 3 < inner.length()) {
                                endCh = (byte) Integer.parseInt(inner.substring(endIdx + 2, endIdx + 4), 16);
                                jj += 4;
                            } else {
                                endCh = (byte) unescChar(inner.charAt(endIdx + 1));
                                jj += 2;
                            }
                        } else {
                            endCh = (byte) inner.charAt(endIdx);
                            jj += 1;
                        }
                        for (int x = Byte.toUnsignedInt(ch); x <= Byte.toUnsignedInt(endCh); x++)
                            chars.add((byte) x);
                    } else {
                        chars.add(ch);
                    }
                }
                res.add(new Rule.Element.CharClass(chars, neg));
                i = end + 1;
                char mod = i < body.length() ? body.charAt(i) : 0;
                if (mod == '*' || mod == '+' || mod == '?') {
                    Rule.Element last = res.removeLast();
                    int min = mod == '+' ? 1 : 0;
                    int max = mod == '?' ? 1 : -1;
                    res.add(new Rule.Element.Repetition(last, min, max));
                    i++;
                }
            }
            else if (c == '.') {
                res.add(new Rule.Element.Dot());
                i++;
                char mod = i < body.length() ? body.charAt(i) : 0;
                if (mod == '*' || mod == '+' || mod == '?') {
                    Rule.Element last = res.removeLast();
                    int min = mod == '+' ? 1 : 0;
                    int max = mod == '?' ? 1 : -1;
                    res.add(new Rule.Element.Repetition(last, min, max));
                    i++;
                }
            }
            else if (c == '|') { res.add(new Rule.Element.Pipe()); i++; }
            else if (Character.isJavaIdentifierStart(c)) {
                int end = i;
                while (end < body.length() && Character.isJavaIdentifierPart(body.charAt(end))) end++;
                String name = body.substring(i, end);
                int rid = rules.getOrDefault(name, 0);
                char next = end < body.length() ? body.charAt(end) : 0;
                Rule.Element.Ref ref = new Rule.Element.Ref(rid);
                if (next == '*') { res.add(new Rule.Element.Repetition(ref, 0, -1)); end++; }
                else if (next == '+') { res.add(new Rule.Element.Repetition(ref, 1, -1)); end++; }
                else if (next == '?') { res.add(new Rule.Element.Repetition(ref, 0, 1)); end++; }
                else res.add(ref);
                i = end;
            }
            else if (c == '(') {
                int end = findMatchingParen(body, i);
                if (end < 0) { i++; continue; }
                List<Rule.Element> inner = parseBody(body.substring(i + 1, end - 1), rules);
                Rule.Element.Group grp = new Rule.Element.Group(inner);
                char next = end < body.length() ? body.charAt(end) : 0;
                if (next == '*') { res.add(new Rule.Element.Repetition(grp, 0, -1)); end++; }
                else if (next == '+') { res.add(new Rule.Element.Repetition(grp, 1, -1)); end++; }
                else if (next == '?') { res.add(new Rule.Element.Repetition(grp, 0, 1)); end++; }
                else res.add(grp);
                i = end;
            }
            else i++;
        }
        return res;
    }

    static int findMatchingBracket(String s, int start) {
        int d = 1;
        for (int j = start + 1; j < s.length(); j++) {
            char c = s.charAt(j);
            if (c == '\\') { j++; continue; }
            if (c == '[') d++;
            else if (c == ']' && --d == 0) return j;
        }
        return -1;
    }

    static int findMatchingParen(String s, int start) {
        int d = 1, end = start + 1;
        while (end < s.length() && d > 0) {
            char c = s.charAt(end);
            if (c == '(') d++;
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

    static char unescChar(char c) { return switch (c) { case 'n' -> '\n'; case 'r' -> '\r'; case 't' -> '\t'; default -> c; }; }

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

    // ---- JSON grammar (GBNF reference, not used at runtime) ----------------

    static final String JSON_GRAMMAR = """
            root ::= object | array
            object ::= "{" ws "}" | "{" ws string ws ":" ws value (ws "," ws string ws ":" ws value)* ws "}"
            array  ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
            value  ::= object | array | string | number | "true" | "false" | "null"
            string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
            number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
            ws     ::= [ \\t\\n\\r]
            """;
}
