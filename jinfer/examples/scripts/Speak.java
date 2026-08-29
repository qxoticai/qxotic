///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-inflect2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Synthesize speech into hello.wav.
//   jbang Speak.java "Local inference, in Java."
import com.qxotic.jinfer.langchain4j.JinferSpeechModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Speak {

    private static final String DEFAULT_MODEL =
            "remixerdec/Inflect-Nano-v2-GGUF:Q8_0";

    public static void main(String[] args) throws IOException {
        String text = args.length > 0 ? args[0] : "Local inference, in Java. No server, no Python.";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var tts = JinferSpeechModel.builder().model(modelRef).build()) {
            byte[] wav = tts.synthesize(text).audio().binaryData();
            Files.write(Path.of("hello.wav"), wav);
            System.out.printf("Wrote hello.wav (%.1f KB).%n", wav.length / 1024.0);
        }
    }
}
