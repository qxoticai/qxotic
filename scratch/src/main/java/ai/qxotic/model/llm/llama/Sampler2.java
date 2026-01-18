package ai.qxotic.model.llm.llama;

import ai.qxotic.model.llm.Sampler;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@FunctionalInterface
public interface Sampler2 extends Sampler<FloatSpan> {

    Sampler2 ARGMAX = fs -> (int) Util.directAccess(fs).argMax(fs);

    default <State> Sampler<State> fromLogits(Function<State, FloatSpan> extractLogits) {
        return state -> applyAsInt(extractLogits.apply(state));
    }

    static Sampler2 selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler2 sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler2.ARGMAX;
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler2 innerSampler;
            if (topp <= 0 || topp >= 1) {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler =
                    logits -> {
                        // apply the temperature to the logits
                        // TODO(peterssen): Derive kernelOps implementation from logits.
                        KernelOps<FloatSpan, FloatMatrixView> logitKernelOps =
                                DefaultKernelOps.getKernelOps();
                        // logits.divideInPlace(0, logits.size(), temperature);
                        logitKernelOps.scale(logits, 1f / temperature, logits);

                        // apply softmax to the logits to get the probabilities for next token
                        logitKernelOps.softMax(logits, logits);
                        return innerSampler.applyAsInt(logits);
                    };
        }
        return sampler;
    }
}
