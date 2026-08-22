///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// Text to speech, fully local: a 4 MB VITS model, no cloud, no ffmpeg.
//   jbang Speak.java "Local inference, in Java."            -> hello.wav
import com.qxotic.jinfer.langchain4j.JinferSpeechModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Speak {
    public static void main(String[] args) throws IOException {
        var text = args.length > 0 ? args[0] : "Local inference, in Java. No server, no Python.";
        try (var tts = JinferSpeechModel.builder().model(Models.speech(args, 1)).build()) {
            byte[] wav = tts.synthesize(text).audio().binaryData();
            Files.write(Path.of("hello.wav"), wav);
            System.out.printf("wrote hello.wav (%.1f KB) - play it with: aplay hello.wav%n", wav.length / 1024.0);
        }
    }
}
