package com.qxotic.jinfer;

public interface PromptCacheSupport {
    long kvBytesPerToken();

    PromptCache create(CacheStore store);
}
