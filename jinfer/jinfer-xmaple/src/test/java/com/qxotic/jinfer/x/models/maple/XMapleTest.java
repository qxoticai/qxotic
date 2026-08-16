package com.qxotic.jinfer.x.models.maple;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.ReplyParser;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class XMapleTest {
    private static final String REF = "hf.co/deepgrove/maple-preview-GGUF:TQ1_0-head-Q4_K";

    @Test
    void loadsPublishedCheckpoint() throws Exception {
        Path path = TestModels.require(REF);
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            Maple model = Maple.loadModel(channel, gguf, Arena.ofAuto());
            assertEquals(24, model.configuration().numberOfLayers());
            assertEquals(256, model.configuration().expertCount());
            assertEquals(8, model.configuration().expertUsedCount());
            assertEquals(512, model.configuration().slidingWindow());
            var reply =
                    ReplyParser.parse(
                            new MapleChatTemplate(model.tokenizer()).parser(model.tokenizer()),
                            SpecialTokens.encode(
                                    model.tokenizer(),
                                    "<think>brief</think><tool_call>"
                                            + "{\"name\":\"ping\",\"arguments\":{}}"
                                            + "</tool_call><|im_end|>"));
            assertInstanceOf(Content.Reasoning.class, reply.content().getFirst());
            assertInstanceOf(Content.ToolCall.class, reply.content().get(1));
            try (Maple.State state = model.newState(2, 1)) {
                model.ingest(state, Batch.step(0));
                var logits = Views.castToSegmentBacked(model.logits(state), "logits");
                int token = Ops.argmax(logits, 0, model.configuration().vocabularySize());
                assertTrue(token >= 0 && token < model.configuration().vocabularySize());
                assertTrue(Float.isFinite(Views.getFloat(logits, token, "logits")));
            }
        }
    }
}
