///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-parakeet
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//DEPS org.slf4j:slf4j-nop:2.0.18

// Speech-to-text on the CPU: transcribe an audio file with NVIDIA Parakeet, with word timing.
//   jbang Transcribe.java speech.wav
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.langchain4j.JinferTranscriptionModel;

import java.nio.file.Files;
import java.nio.file.Path;

public class Transcribe {

    private static final String DEFAULT_MODEL =
            "mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf";

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("usage: Transcribe <audio> [model-ref]");
            System.exit(2);
        }
        Path audio = Path.of(args[0]);
        if (!Files.isRegularFile(audio)) {
            System.err.println("no such file: " + audio);
            System.exit(2);
        }
        String model = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var transcriber = JinferTranscriptionModel.builder().model(model).build()) {
            Transcription transcription = transcriber.transcribe(audio);
            System.out.println(transcription.text());
            for (Transcription.Token token : transcription.tokens()) {
                System.err.printf("%6.2f-%6.2f %s%n", token.start(), token.end(), token.text());
            }
        }
    }
}
