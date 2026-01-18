package ai.qxotic.model.llm.mistral;

import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.StandardTokenType;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.AbstractTokenizer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Byte Pair Encoding tokenizer.
 *
 * <p>Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows
 * along the <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2
 * tokenizer</a>
 */
public class MistralTokenizer extends AbstractTokenizer {

    private final int byte0;

    @Override
    public VocabularyImplWithScores vocabulary() {
        return (VocabularyImplWithScores) super.vocabulary();
    }

    public MistralTokenizer(
            VocabularyImplWithScores vocabulary, Normalizer normalizer, TextSplitter preTokenizer) {
        super(vocabulary, normalizer, preTokenizer);
        this.byte0 = vocabulary.id("<0x00>");
    }

    @Override
    public IntSequence encode(String text) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        return decode(tokens).getBytes(StandardCharsets.UTF_8);
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {

        List<Integer> tokens = new ArrayList<>();

        // first encode every individual codepoint in the input string
        text.codePoints()
                .forEachOrdered(
                        cpi -> {
                            String singleCodepoint = Character.toString(cpi);
                            int id =
                                    vocabulary.contains(singleCodepoint)
                                            ? vocabulary.id(singleCodepoint)
                                            : -1;

                            if (id != -1) {
                                // we found this codepoint in vocab, add it as a token
                                tokens.add(id);
                            } else {
                                // byte_fallback encoding: just encode each byte as a token
                                // +byte0 here to skip all the control and special tokens e.g.
                                // <unk>, <s>, </s>
                                // so the individual bytes only start at token <0x00>
                                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                                }
                            }
                        });

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer =
                        vocabulary.token(tokens.get(i)) + vocabulary.token(tokens.get(i + 1));
                int id = vocabulary.contains(str_buffer) ? vocabulary.id(str_buffer) : -1;
                if (id != -1 && vocabulary().getScore(id) > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocabulary().getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return IntSequence.wrap(tokens);
    }

    @Override
    public String decode(IntSequence tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.token(token);
            if (vocabulary.isTokenOfType(token, StandardTokenType.BYTE)) {
                // some tokens designate raw bytes e.g. '<0x10>'
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6
                        && tokenString.startsWith(prefix)
                        && tokenString.endsWith(suffix)) {
                    String code =
                            tokenString.substring(
                                    prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('▁', ' ');
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }
}
