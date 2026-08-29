///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Constrain generation to a JSON shape with a GBNF grammar.
//   jbang Json.java "Ada Lovelace, born 1815 in London, wrote the first algorithm."
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferChatRequestParameters;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;

public class Json {

    private static final String DEFAULT_MODEL =
            "unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0";

    private static final String GRAMMAR = """
            root ::= "{" ws "\\"name\\":" ws str "," ws "\\"year\\":" ws num "," ws "\\"city\\":" ws str ws "}"
            str  ::= "\\"" [^"]* "\\""
            num  ::= [0-9]+
            ws   ::= " "*
            """;

    public static void main(String[] args) {
        String text = args.length > 0 ? args[0]
                : "Ada Lovelace, born 1815 in London, wrote the first algorithm.";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var model = JinferChatModel.builder()
                .model(modelRef)
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
