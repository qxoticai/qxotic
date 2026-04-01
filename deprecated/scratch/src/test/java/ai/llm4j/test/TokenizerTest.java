package ai.llm4j.test;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.toknroll.Tokenizer;
import java.util.stream.Stream;

public abstract class TokenizerTest {

    public static Stream<Tokenizer> tokenizerProvider() {
        return Stream.of(CLASSIC_TOKENIZER, TIKTOKEN_TOKENIZER);
    }

    public static Stream<ChatFormat> chatFormatProvider() {
        throw new UnsupportedOperationException();
        // return tokenizerProvider().map(Llama3ChatFormat::new);
    }

    static final Tokenizer CLASSIC_TOKENIZER = loadClassicTokenizer();
    static final Tokenizer TIKTOKEN_TOKENIZER = loadTiktokenTokenizer();

    static Tokenizer loadClassicTokenizer() {
        throw new UnsupportedOperationException();
        //        String tokenizerPath = System.getenv("TOKENIZER_PATH");
        //        if (tokenizerPath == null) {
        //            // throw new IllegalStateException("TOKENIZER_PATH environment variable must
        // be set");
        //            tokenizerPath =
        // "/home/mukel/Desktop/playground/models/mukel/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf";
        //        }
        //        try {
        //            return Llama3ModelLoader.loadTokenizer(GGUF.read(Path.of(tokenizerPath)));
        //            //return
        // Tiktoken.loadLlama3(Path.of("/home/mukel/Downloads/llama3.tiktoken"));
        //        } catch (IOException e) {
        //            throw new RuntimeException(e);
        //        }
    }

    static Tokenizer loadTiktokenTokenizer() {

        throw new UnsupportedOperationException();
        //        String tokenizerPath = System.getenv("TOKENIZER_PATH");
        //        if (tokenizerPath == null) {
        //            // throw new IllegalStateException("TOKENIZER_PATH environment variable must
        // be set");
        //            tokenizerPath =
        // "/home/mukel/Desktop/playground/models/mukel/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf";
        //        }
        //        try {
        //            return
        // Llama3ModelLoader.loadTokenizerFromTiktoken(Path.of("/home/mukel/Downloads/llama3.tiktoken"));
        //        } catch (IOException e) {
        //            throw new RuntimeException(e);
        //        }
    }
}
