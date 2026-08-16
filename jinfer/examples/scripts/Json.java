///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// Structured output that CANNOT be malformed: a GBNF grammar constrains SAMPLING, so the model is
// unable to emit a token that would break the schema. No retries, no "please reply in JSON".
//   jbang Json.java "Ada Lovelace, born 1815 in London, wrote the first algorithm."
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferChatRequestParameters;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;

public class Json {

    // {"name": "...", "year": 1234, "city": "..."} - in this order, nothing else possible.
    static final String GRAMMAR = """
            root ::= "{" ws "\\"name\\":" ws str "," ws "\\"year\\":" ws num "," ws "\\"city\\":" ws str ws "}"
            str  ::= "\\"" [^"]* "\\""
            num  ::= [0-9]+
            ws   ::= " "*
            """;

    public static void main(String[] args) {
        var text = args.length > 0 ? args[0]
                : "Ada Lovelace, born 1815 in London, wrote the first algorithm.";
        try (var model = JinferChatModel.builder()
                .model(Models.chat(args, 1))
                .maxOutputTokens(96)
                .thinking(false)
                .build()) {
            var request = ChatRequest.builder()
                    .messages(UserMessage.from("Extract the person as JSON:\n" + text))
                    .parameters(JinferChatRequestParameters.builder().grammar(GRAMMAR).build())
                    .build();
            System.out.println(model.chat(request).aiMessage().text());
        }
    }
}
