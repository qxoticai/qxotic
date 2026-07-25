package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.Set;

/**
 * The two-tokenization-domain law as a builder: templates lower a conversation to token runs where
 * scaffolding is TRUSTED (special ids, template-authored text whose marker spellings mint ids) and
 * conversation content is PLAIN (never special-aware, so content can never mint control tokens).
 *
 * <p>Contiguity is the load-bearing invariant: adjacent text - whatever mix of {@link #text} and
 * the plain stretches of {@link #trusted} - accumulates into ONE {@link Tokenizer#encode} run,
 * exactly how a rendered template tokenizes (specials force the only splits, BPE merges across
 * every other boundary). The final flush is {@link #build}'s job, so a template cannot forget it.
 *
 * <p>{@link #trusted} scans template-authored text for the tokenizer's special spellings (the same
 * longest-match, prefix-free set the render+rescan oracle uses - {@link SpecialTokens#encoder}) and
 * mints their ids in place; the surrounding stretches stay part of the neighbouring plain run. This
 * is what makes a port's fixed instruction blocks byte-exact with the Jinja render without
 * hand-splitting the constants at each spelling.
 */
public final class TokenRuns {

    private final Tokenizer tokenizer;
    private final Set<String> spellings; // the rescan's kept special spellings
    private IntSequence.Builder ids = IntSequence.newBuilder(); // replaced at cuts
    private final StringBuilder run = new StringBuilder();
    private final java.util.List<Batch> out = new java.util.ArrayList<>(); // atomic-block streams

    public TokenRuns(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.spellings = SpecialTokens.encoder(tokenizer).tokens();
    }

    private TokenRuns(Tokenizer tokenizer, Set<String> spellings) {
        this.tokenizer = tokenizer;
        this.spellings = spellings;
    }

    /** A fresh builder sharing this one's compiled spelling set (one scan table per template). */
    public TokenRuns fresh() {
        return new TokenRuns(tokenizer, spellings);
    }

    /** One trusted special id (role markers, span delimiters). */
    public TokenRuns id(int id) {
        flush();
        ids.add(id);
        return this;
    }

    /** Conversation content: plain-encoded, joins the current run, can never mint a special. */
    public TokenRuns text(String s) {
        run.append(s);
        return this;
    }

    /**
     * Template-authored text: marker spellings mint their ids (matching the render+rescan), the
     * stretches between join the neighbouring plain runs.
     */
    public TokenRuns trusted(String s) {
        int i = 0;
        while (i < s.length()) {
            int at = s.length();
            String hit = null;
            for (String spelling : spellings) {
                int found = s.indexOf(spelling, i);
                if (found >= 0
                        && (found < at || (found == at && spelling.length() > hit.length()))) {
                    at = found;
                    hit = spelling;
                }
            }
            run.append(s, i, at);
            if (hit == null) break;
            id(tokenizer.vocabulary().id(hit));
            i = at + hit.length();
        }
        return this;
    }

    /** Splices generated ids verbatim (model-exact bytes for echoed payloads). */
    public TokenRuns verbatim(IntSequence v) {
        flush();
        ids.addAll(v);
        return this;
    }

    /**
     * Splices an ATOMIC non-token block - a media {@link Batch#embeddings} splice - cutting the id
     * stream around it. The block's attention semantics (bidirectional image group vs causal audio)
     * live on the batch itself, and {@link Batch#prepare} guarantees embedding blocks are never
     * split; this builder only sequences.
     */
    public TokenRuns block(Batch block) {
        cut();
        out.add(block);
        return this;
    }

    /** Flushes the pending run and returns everything as one id sequence (token-only streams). */
    public IntSequence build() {
        if (!out.isEmpty()) {
            throw new IllegalStateException("stream contains atomic blocks: use batches()");
        }
        flush();
        return ids.build();
    }

    /** {@link #build} as one prefill batch - the common whole-conversation shape. */
    public Batch batch() {
        return Batch.prefill(build().toArray());
    }

    /** The full stream as batches: token runs cut around every atomic block, final flush owned. */
    public java.util.List<Batch> batches() {
        cut();
        return java.util.List.copyOf(out);
    }

    private void cut() {
        flush();
        IntSequence pending = ids.build();
        if (pending.length() > 0) {
            out.add(Batch.prefill(pending.toArray()));
            ids = IntSequence.newBuilder();
        }
    }

    private void flush() {
        if (run.isEmpty()) return;
        ids.addAll(tokenizer.encode(run.toString()));
        run.setLength(0);
    }
}
