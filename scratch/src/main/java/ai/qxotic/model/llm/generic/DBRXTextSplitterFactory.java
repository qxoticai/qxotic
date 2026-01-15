package ai.qxotic.model.llm.generic;

import com.google.auto.service.AutoService;
import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.model.llm.llama.LlamaBPETextSplitterFactory;

@AutoService(TextSplitterFactory.class)
public class DBRXTextSplitterFactory extends LlamaBPETextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "dbrx";
    }
}
