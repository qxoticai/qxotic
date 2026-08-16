///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-xlangchain4j:0.1.0
//SOURCES Models.java

// Streaming chat, in-process. No server, no Python, no JNI glue.
//   jbang Chat.java "Explain HTTP/3 in two sentences."
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.concurrent.CompletableFuture;

public class Chat {
    public static void main(String[] args) {
        var prompt = args.length > 0 ? args[0] : "Explain HTTP/3 in two sentences.";
        try (var model = JinferChatModel.builder().model(Models.chat(args, 1)).build()) {
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
