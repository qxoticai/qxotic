package ai.qxotic.model.llm;

import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;

// TokenizerFactory interface that implementations will need to implement
public interface TokenizerFactory<Source> {

    String getSourceName();

    String getTokenizerName();

    Tokenizer createTokenizer(Source source, Normalizer normalizer, TextSplitter textSplitter);
}
