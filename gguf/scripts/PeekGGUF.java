import com.qxotic.format.gguf.GGUF;
import java.net.URL;
import java.nio.channels.Channels;
import java.io.BufferedInputStream;

public class PeekGGUF {
    public static void main(String[] args) throws Exception {
        String url = args[0];
        try (var stream = new BufferedInputStream(new URL(url).openStream());
             var channel = Channels.newChannel(stream)) {
            GGUF gguf = GGUF.read(channel);
            System.out.println(gguf.toString(true, true));
        }
    }
}
