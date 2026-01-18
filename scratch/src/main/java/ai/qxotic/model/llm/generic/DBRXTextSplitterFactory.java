package ai.qxotic.model.llm.generic;

import ai.qxotic.model.llm.TextSplitterFactory;
import ai.qxotic.model.llm.llama.LlamaBPETextSplitterFactory;
import com.google.auto.service.AutoService;

@AutoService(TextSplitterFactory.class)
public class DBRXTextSplitterFactory extends LlamaBPETextSplitterFactory {
    @Override
    public String getTextSplitterName() {
        return "dbrx";
    }
}
