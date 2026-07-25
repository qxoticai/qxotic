package com.qxotic.jinfer.spring.ai.autoconfigure;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Flat properties for {@link JinferChatAutoConfiguration}, bound under {@code
 * spring.ai.jinfer.chat}.
 */
@ConfigurationProperties("spring.ai.jinfer.chat")
public class JinferChatProperties {

    /** Path to the GGUF model file (required). */
    private String modelPath;

    /** Path to the media sidecar (mmproj GGUF: vision/audio encoders) for multimodal models. */
    private String mediaProjector;

    /** Path to a cached-prompt artifact (.jkv) to mount at startup; model-seed-checked. */
    private String cachedPrompts;

    /**
     * Live conversation states kept resident and reused append-only when a request's conversation
     * strictly extends one (the multi-turn zero-restore tier). 0 (default) disables the pool.
     */
    private int cachedSessions;

    /** Context window; 0 = the model's own maximum. */
    private int contextLength;

    private Double temperature;
    private Double topP;
    private Integer maxTokens;
    private Long seed;

    /** The model's reasoning scaffold toggle (templates without one ignore it). Default on. */
    private Boolean thinking;

    /** Wall-clock generation deadline; null = none. */
    private Duration timeout;

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public String getMediaProjector() {
        return mediaProjector;
    }

    public void setMediaProjector(String mediaProjector) {
        this.mediaProjector = mediaProjector;
    }

    public String getCachedPrompts() {
        return cachedPrompts;
    }

    public void setCachedPrompts(String cachedPrompts) {
        this.cachedPrompts = cachedPrompts;
    }

    public int getCachedSessions() {
        return cachedSessions;
    }

    public void setCachedSessions(int cachedSessions) {
        this.cachedSessions = cachedSessions;
    }

    public int getContextLength() {
        return contextLength;
    }

    public void setContextLength(int contextLength) {
        this.contextLength = contextLength;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Double getTopP() {
        return topP;
    }

    public void setTopP(Double topP) {
        this.topP = topP;
    }

    public Integer getMaxTokens() {
        return maxTokens;
    }

    public void setMaxTokens(Integer maxTokens) {
        this.maxTokens = maxTokens;
    }

    public Long getSeed() {
        return seed;
    }

    public void setSeed(Long seed) {
        this.seed = seed;
    }

    public Boolean getThinking() {
        return thinking;
    }

    public void setThinking(Boolean thinking) {
        this.thinking = thinking;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public void setTimeout(Duration timeout) {
        this.timeout = timeout;
    }
}
