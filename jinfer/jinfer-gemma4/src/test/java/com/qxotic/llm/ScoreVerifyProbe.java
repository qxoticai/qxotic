// Decisive isolation: is Batch.score (multi-row ALL) row-0 argmax == single-step argmax at the same
// context, independent of MTP? Greedy-decode two ways in lockstep and report the first confident
// disagreement. If they diverge confidently, the multi-row verify path itself is the culprit.
package com.qxotic.llm;
import com.qxotic.jinfer.Batch;
import java.nio.file.Path;
import java.util.*;
public class ScoreVerifyProbe {
  public static void main(String[] a) throws Exception {
    Gemma4 m = Gemma4.loadModel(Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf"), 4096);
    var tk=m.tokenizer(); int bos=tk.getSpecialTokens().getOrDefault("<bos>",2); var stops=m.stopTokens(); int V=m.config().vocabularySize();
    for (String prompt : new String[]{"def fibonacci(n):", "Once upon a time, in a quiet village by the mountains,"}) {
      List<Integer> e=new ArrayList<>(); e.add(bos); e.addAll(tk.encode(prompt));
      int[] ids=e.stream().mapToInt(Integer::intValue).toArray();
      // reference: pure single-step greedy
      Gemma4.State ps=m.newState(4096,256); m.ingest(ps,Batch.prefill(ids));
      List<Integer> step=new ArrayList<>(); int t=m.logits(ps,0).argmax(0,V);
      for(int i=0;i<60 && !stops.contains(t);i++){step.add(t);m.ingest(ps,Batch.step(t));t=m.logits(ps,0).argmax(0,V);}
      // candidate: greedy where each token is decided by a 2-ROW score batch (row 0), like MTP verify
      Gemma4.State qs=m.newState(4096,256); m.ingest(qs,Batch.prefill(ids));
      List<Integer> score=new ArrayList<>();
      int prev = m.logits(qs,0).argmax(0,V);   // first token via prefill (same as step)
      for(int i=0;i<60 && !stops.contains(prev);i++){
        score.add(prev);
        // verify-shape: ingest [prev, dummy] as a score batch, read row 0 (after prev), keep only prev
        int base=qs.position();
        m.ingest(qs, Batch.score(new int[]{prev, prev}));
        int nxt=m.logits(qs,0).argmax(0,V);
        qs.resumeAt(base+1);                    // keep only prev (drop the dummy), exactly like the loop
        prev=nxt;
      }
      int d=0; while(d<Math.min(step.size(),score.size()) && step.get(d).equals(score.get(d))) d++;
      System.out.printf("%-20s step==score to %d/%d%n", prompt.substring(0,Math.min(18,prompt.length())), d, step.size());
      if (d<step.size() && d<score.size()) System.out.printf("   at %d: step=%d score=%d%n", d, step.get(d), score.get(d));
    }
  }
}
