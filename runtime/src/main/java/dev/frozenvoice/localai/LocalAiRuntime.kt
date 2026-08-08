package dev.frozenvoice.localai

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/** A model-independent generation policy. The native sampler is configured from this value. */
enum class GenerationProfile(
    val defaultMaxNewTokens: Int,
    val temperature: Float,
    val topP: Float,
    val topK: Int,
    val repeatPenalty: Float,
) {
    CHAT(
        defaultMaxNewTokens = 256,
        temperature = 0.7f,
        topP = 0.9f,
        topK = 40,
        repeatPenalty = 1.08f,
    ),
    STRUCTURED(
        defaultMaxNewTokens = 96,
        temperature = 0.0f,
        topP = 1.0f,
        topK = 40,
        repeatPenalty = 1.0f,
    ),
    TOOL_SELECTION(
        defaultMaxNewTokens = 96,
        temperature = 0.0f,
        topP = 1.0f,
        topK = 40,
        repeatPenalty = 1.0f,
    ),
}

enum class ConversationMode {
    CONTINUE,
    RESET,
    STATELESS,
}

data class GenerationRequest(
    val prompt: String,
    val profile: GenerationProfile = GenerationProfile.CHAT,
    val maxNewTokens: Int = profile.defaultMaxNewTokens,
    val grammar: String? = null,
    val conversationMode: ConversationMode = ConversationMode.CONTINUE,
    /** Optional system message used when this request starts or resets a conversation. */
    val systemPrompt: String? = null,
) {
    init {
        require(maxNewTokens in 1..MAX_NEW_TOKENS) {
            "maxNewTokens must be in 1..$MAX_NEW_TOKENS."
        }
        require(prompt.isNotBlank()) { "prompt must not be blank." }
        require(grammar == null || grammar.isNotBlank()) { "grammar must be null or non-blank." }
        require(systemPrompt == null || systemPrompt.isNotBlank()) {
            "systemPrompt must be null or non-blank."
        }
    }

    internal fun resolvedPrompt(): String = prompt

    internal fun resolvedSystemPrompt(): String? = systemPrompt

    private companion object {
        const val MAX_NEW_TOKENS = 4_096
    }
}

data class RuntimeOptions(
    val contextSize: Int = 8_192,
    val batchSize: Int = 512,
    val threadCount: Int? = null,
    val batchThreadCount: Int? = null,
) {
    init {
        require(contextSize >= 256) { "contextSize must be at least 256." }
        require(batchSize in 1..contextSize) { "batchSize must fit within contextSize." }
        require(threadCount == null || threadCount > 0) { "threadCount must be positive." }
        require(batchThreadCount == null || batchThreadCount > 0) {
            "batchThreadCount must be positive."
        }
    }
}

data class RuntimeSettings(
    val contextSize: Int,
    val batchSize: Int,
    val microBatchSize: Int,
    val threadCount: Int,
    val batchThreadCount: Int,
    val backend: String,
)

data class ModelInfo(
    val filePath: String,
    val fileName: String,
    val fileSizeBytes: Long,
    val modelDescription: String?,
    val architecture: String?,
    val trainedContextLength: Int?,
    val runtimeContextLength: Int,
    val parameterCount: Long?,
    val hasEmbeddedChatTemplate: Boolean,
    val loadTimeMillis: Long,
)

enum class TerminationReason {
    EOS,
    MAX_TOKENS,
    STOP_SEQUENCE,
    CANCELLED,
    ERROR,
    UNKNOWN,
}

data class InferenceMetrics(
    val modelLoadMillis: Long,
    val timeToFirstTokenMillis: Long,
    val totalGenerationMillis: Long,
    val promptTokenCount: Int,
    val generatedTokenCount: Int,
    val promptProcessingMillis: Long,
    val generationMillis: Long,
    val promptTokensPerSecond: Double,
    val generationTokensPerSecond: Double,
    val modelReused: Boolean,
    val promptPrefixReused: Boolean,
    val runtimeContextLength: Int,
    val requestedMaxNewTokens: Int,
    val profile: GenerationProfile,
    val terminationReason: TerminationReason,
    val runtimeSettings: RuntimeSettings,
)

sealed interface RuntimeState {
    data object Uninitialized : RuntimeState
    data object Initializing : RuntimeState
    data object Initialized : RuntimeState
    data object LoadingModel : RuntimeState
    data object ModelReady : RuntimeState
    data object ProcessingPrompt : RuntimeState
    data object Generating : RuntimeState
    data object UnloadingModel : RuntimeState
    data class Error(val exception: LocalAiRuntimeException) : RuntimeState
}

open class LocalAiRuntimeException(
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause)

/** Android-facing API. No llama.cpp type is exposed here. */
interface LocalAiRuntime {
    val state: StateFlow<RuntimeState>

    suspend fun loadModel(path: String): ModelInfo

    /** Streams UTF-8 text pieces and leaves the last metrics available via [lastMetrics]. */
    fun generate(request: GenerationRequest): Flow<String>

    suspend fun generateStructured(request: GenerationRequest): GenerationResult

    suspend fun setSystemPrompt(systemPrompt: String)

    fun cancel()

    suspend fun resetConversation(systemPrompt: String? = null)

    suspend fun unload()

    fun modelInfo(): ModelInfo?

    fun lastMetrics(): InferenceMetrics?
}

data class GenerationResult(
    val text: String,
    val metrics: InferenceMetrics,
)
