///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// Streaming chat, in-process. No server, no Python, no JNI glue.
//   jbang Chat.java "Explain HTTP/3 in two sentences."
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.model.chat.response.*;
import java.util.concurrent.CountDownLatch;

public class Chat {
    public static void main(String[] args) throws Exception {
        var prompt = args.length > 0 ? args[0] : "Explain HTTP/3 in two sentences.";
        try (var model = JinferChatModel.builder().modelPath(Models.chat(args, 1)).build()) {
            var done = new CountDownLatch(1);
            model.streaming().chat(prompt, new StreamingChatResponseHandler() {
                public void onPartialResponse(String token) { System.out.print(token); System.out.flush(); }
                public void onCompleteResponse(ChatResponse r) { System.out.println(); done.countDown(); }
                public void onError(Throwable t) { t.printStackTrace(); done.countDown(); }
            });
            done.await();
        }
    }
}
