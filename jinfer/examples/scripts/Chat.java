///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Stream a local model directly to the terminal.
//   jbang Chat.java "Explain HTTP/3 in two sentences."
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

import java.util.concurrent.CompletableFuture;

public class Chat {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0";

    public static void main(String[] args) {
        String prompt = args.length > 0 ? args[0] : "Explain HTTP/3 in two sentences.";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var model = JinferChatModel.builder().model(modelRef).build()) {
            var done = new CompletableFuture<Void>();
            model.streaming()
                    .chat(
                            prompt,
                            new StreamingChatResponseHandler() {
                                @Override
                                public void onPartialResponse(String text) {
                                    System.out.print(text);
                                    System.out.flush();
                                }

                                @Override
                                public void onCompleteResponse(ChatResponse response) {
                                    System.out.println();
                                    done.complete(null);
                                }

                                @Override
                                public void onError(Throwable error) {
                                    done.completeExceptionally(error);
                                }
                            });
            done.join();
        }
    }
}
