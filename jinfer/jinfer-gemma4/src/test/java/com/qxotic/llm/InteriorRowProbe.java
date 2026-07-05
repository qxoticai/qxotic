// Is logits(s, output>0) on a multi-row score batch (a) reproducible run-to-run and (b) equal to the
// single-step decode of the same prefix? The MTP accept loop reads these interior rows; ScoreVerifyProbe
// only read row 0. This isolates interior-row correctness/determinism.
package com.qxotic.llm;
import com.qxotic.jinfer.Batch;
import java.nio.file.Path;
import java.util.*;
public class InteriorRowProbe {
  public static void main(String[] a) throws Exception {
    Gemma4 m = Gemma4.loadModel(Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf"), 4096);
    var tk=m.tokenizer(); int bos=tk.getSpecialTokens().getOrDefault("<bos>",2); int V=m.config().vocabularySize();
    List<Integer> e=new ArrayList<>(); e.add(bos); e.addAll(tk.encode("def fibonacci(n):"));
    int[] ids=e.stream().mapToInt(Integer::intValue).toArray();
    // pick two real consecutive tokens t0,t1 via greedy so the batch is in-distribution
    Gemma4.State g=m.newState(4096,64); m.ingest(g,Batch.prefill(ids));
    int t0=m.logits(g,0).argmax(0,V); m.ingest(g,Batch.step(t0)); int t1=m.logits(g,0).argmax(0,V);
    // single-step reference: prefill ids, step t0, step t1, read logits(0) -> token after t1
    Gemma4.State a1=m.newState(4096,64); m.ingest(a1,Batch.prefill(ids)); m.ingest(a1,Batch.step(t0)); m.ingest(a1,Batch.step(t1));
    int refStep=m.logits(a1,0).argmax(0,V);
    // score batch: prefill ids, then score([t0,t1,t1]); interior row 1 = after [t0,t1] -> should == refStep
    int[] r0=new int[3], r1=new int[3];
    for(int rep=0;rep<3;rep++){
      Gemma4.State b=m.newState(4096,64); m.ingest(b,Batch.prefill(ids));
      m.ingest(b, Batch.score(new int[]{t0,t1,t1}));
      r0[rep]=m.logits(b,0).argmax(0,V);   // after t0 -> should == t1
      r1[rep]=m.logits(b,1).argmax(0,V);   // after [t0,t1] -> should == refStep
    }
    System.out.printf("t0=%d t1=%d refStep(single)=%d%n",t0,t1,refStep);
    System.out.printf("score row0 (after t0, want %d): %s%n", t1, Arrays.toString(r0));
    System.out.printf("score row1 (after t0,t1, want %d): %s%n", refStep, Arrays.toString(r1));
    boolean row1ok = r1[0]==refStep && r1[0]==r1[1] && r1[1]==r1[2];
    System.out.println(row1ok ? "INTERIOR ROW OK" : "INTERIOR ROW BROKEN (retention bug or nondeterministic)");
  }
}
