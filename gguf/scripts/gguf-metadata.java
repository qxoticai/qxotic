///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS com.qxotic:gguf:0.1.0
//DEPS info.picocli:picocli:4.7.7

import com.qxotic.format.gguf.GGUF;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;

/**
 * GGUF metadata extractor - extracts only the metadata bytes from a GGUF file.
 * Writes raw bytes from 0 to tensorDataOffset.
 */
@Command(
    name = "gguf-metadata",
    description = "Extract GGUF metadata bytes to a separate file",
    mixinStandardHelpOptions = true
)
class ggufMetadata implements Callable<Integer> {

    @CommandLine.Parameters(paramLabel = "SOURCE", description = "File path or URL to GGUF file")
    String source;

    @CommandLine.Option(names = {"-o", "--output"}, description = "Output file path")
    String output;

    @Override
    public Integer call() throws Exception {
        URL url = source.contains("://") ? new URL(source) : new URL("file", "", source);
        
        String outputPath = output;
        if (outputPath == null) {
            String name = source.replaceAll(".*/", "").replaceAll("\\?.*", "");
            outputPath = name + ".gguf.metadata";
        }
        
        System.err.println("Reading: " + url);
        
        // First pass: find tensor data offset
        long tensorDataOffset;
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream(), 1 << 16))) {
            GGUF gguf = GGUF.read(channel);
            tensorDataOffset = gguf.getTensorDataOffset();
            System.err.println("Metadata ends at offset: " + tensorDataOffset);
        }
        
        // Second pass: copy raw metadata bytes
        System.err.println("Writing: " + outputPath);
        try (var in = Channels.newChannel(new BufferedInputStream(url.openStream(), 1 << 16));
             var out = new BufferedOutputStream(new FileOutputStream(outputPath))) {
            
            byte[] buffer = new byte[(int) tensorDataOffset];
            var bb = java.nio.ByteBuffer.wrap(buffer);
            in.read(bb);
            out.write(buffer);
        }
        
        System.err.println("Done! Metadata saved to: " + outputPath);
        return 0;
    }

    public static void main(String... args) {
        System.exit(new CommandLine(new ggufMetadata()).execute(args));
    }
}
