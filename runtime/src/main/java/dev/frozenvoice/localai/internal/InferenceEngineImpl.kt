package dev.frozenvoice.localai.internal

import android.content.Context
import dalvik.annotation.optimization.FastNative
import dev.frozenvoice.localai.ConversationMode
import dev.frozenvoice.localai.GenerationProfile
import dev.frozenvoice.localai.GenerationRequest
import dev.frozenvoice.localai.InferenceMetrics
import dev.frozenvoice.localai.LocalAiRuntime
import dev.frozenvoice.localai.LocalAiRuntimeException
import dev.frozenvoice.localai.ModelInfo
import dev.frozenvoice.localai.RuntimeOptions
import dev.frozenvoice.localai.RuntimeSettings
import dev.frozenvoice.localai.RuntimeState
import dev.frozenvoice.localai.TerminationReason
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

internal class InferenceEngineImpl(
    context: Context,
    options: RuntimeOptions,
) : LocalAiRuntime {
    private val session = SerializedRuntimeSession(
        nativeLibraryDir = requireNotNull(context.applicationContext ?: context)
            .applicationInfo.nativeLibraryDir,
        options = options,
        bridge = JniNativeRuntimeBridge(),
    )

    override val state: StateFlow<RuntimeState> = session.state

    override suspend fun loadModel(path: String): ModelInfo = session.loadModel(path)

    override fun generate(request: GenerationRequest): Flow<String> = session.generate(request)

    override suspend fun generateStructured(
        request: GenerationRequest,
    ): dev.frozenvoice.localai.GenerationResult = session.generateStructured(request)

    override suspend fun setSystemPrompt(systemPrompt: String) {
        session.resetConversation(systemPrompt)
    }

    override fun cancel() = session.cancel()

    override suspend fun resetConversation(systemPrompt: String?) = session.resetConversation(systemPrompt)

    override suspend fun unload() = session.unload()

    override fun modelInfo(): ModelInfo? = session.modelInfo()

    override fun lastMetrics(): InferenceMetrics? = session.lastMetrics()
}

@OptIn(ExperimentalCoroutinesApi::class)
internal class SerializedRuntimeSession(
    private val nativeLibraryDir: String,
    private val options: RuntimeOptions,
    private val bridge: NativeRuntimeBridge,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO.limitedParallelism(1),
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private val operationMutex = Mutex()
    private val mutableState = MutableStateFlow<RuntimeState>(RuntimeState.Uninitialized)
    val state: StateFlow<RuntimeState> = mutableState.asStateFlow()

    private var initialized = false
    private var loadedModelPath: String? = null
    private var loadedModel: ModelInfo? = null
    private var loadedRuntimeSettings = RuntimeSettings(0, 0, 0, 0, 0, "unknown")
    private var activeSystemPrompt: String? = null
    private var lastLoadMillis = 0L
    private var lastModelWasReused = false

    @Volatile
    private var latestMetrics: InferenceMetrics? = null
    private val cancellationRequested = AtomicBoolean(false)

    suspend fun loadModel(path: String): ModelInfo = withContext(dispatcher) {
        operationMutex.withLock {
            try {
                ensureInitialized()
                val canonicalPath = validateModelPath(path)
                if (canonicalPath == loadedModelPath && loadedModel != null) {
                    lastLoadMillis = 0L
                    lastModelWasReused = true
                    mutableState.value = RuntimeState.ModelReady
                    return@withLock loadedModel!!.copy(loadTimeMillis = 0L)
                }

                mutableState.value = RuntimeState.LoadingModel
                unloadLocked()
                val startedAt = nanoTime()
                bridge.load(canonicalPath)
                bridge.prepare()
                loadedRuntimeSettings = bridge.runtimeSettings()
                val nativeInfo = bridge.modelInfo()
                val loadMillis = elapsedMillis(startedAt)
                val info = ModelInfo(
                    filePath = canonicalPath,
                    fileName = File(canonicalPath).name,
                    fileSizeBytes = File(canonicalPath).length(),
                    modelDescription = nativeInfo.description,
                    architecture = nativeInfo.architecture,
                    trainedContextLength = nativeInfo.trainedContextLength,
                    runtimeContextLength = loadedRuntimeSettings.contextSize,
                    parameterCount = nativeInfo.parameterCount,
                    hasEmbeddedChatTemplate = nativeInfo.hasEmbeddedChatTemplate,
                    loadTimeMillis = loadMillis,
                )
                loadedModelPath = canonicalPath
                loadedModel = info
                lastLoadMillis = loadMillis
                lastModelWasReused = false
                activeSystemPrompt = null
                cancellationRequested.set(false)
                mutableState.value = RuntimeState.ModelReady
                info
            } catch (failure: Throwable) {
                runCatching { if (initialized) bridge.unload() }
                loadedModelPath = null
                loadedModel = null
                activeSystemPrompt = null
                loadedRuntimeSettings = RuntimeSettings(0, 0, 0, 0, 0, "unknown")
                latestMetrics = null
                cancellationRequested.set(true)
                val error = failure.asRuntimeError("Model loading failed")
                mutableState.value = RuntimeState.Error(error)
                throw error
            }
        }
    }

    fun generate(request: GenerationRequest): Flow<String> = flow {
        val prompt = request.resolvedPrompt()
        val systemPrompt = request.resolvedSystemPrompt()
        var stateless = false
        try {
            operationMutex.withLock {
                checkReady()
                stateless = request.conversationMode == ConversationMode.STATELESS
                cancellationRequested.set(false)
                prepareConversationLocked(
                    systemPrompt = systemPrompt,
                    mode = request.conversationMode,
                )

                mutableState.value = RuntimeState.ProcessingPrompt
                val result = bridge.processUserPrompt(
                    prompt = prompt,
                    profile = request.profile,
                    maxNewTokens = request.maxNewTokens,
                    grammar = request.grammar,
                )
                if (result != 0) {
                    throw LocalAiRuntimeException(
                        "Prompt processing failed with native code $result.",
                    )
                }

                mutableState.value = RuntimeState.Generating
                while (!cancellationRequested.get()) {
                    val token = bridge.generateNextToken() ?: break
                    if (token.isNotEmpty()) emit(token)
                }

                latestMetrics = buildMetrics(request)
                mutableState.value = RuntimeState.ModelReady
                if (stateless) {
                    // Do this after reading metrics; native reset intentionally keeps the
                    // last inference counters available to the caller.
                    bridge.resetConversation()
                    activeSystemPrompt = null
                }
            }
        } catch (cancelled: CancellationException) {
            cancellationRequested.set(true)
            bridge.requestCancel()
            runCatching {
                withContext(NonCancellable + dispatcher) {
                    operationMutex.withLock {
                        if (loadedModelPath != null) {
                            latestMetrics = buildMetrics(request)
                            if (stateless) {
                                bridge.resetConversation()
                                activeSystemPrompt = null
                            }
                        }
                        mutableState.value = RuntimeState.ModelReady
                    }
                }
            }
            throw cancelled
        } catch (failure: Throwable) {
            cancellationRequested.set(true)
            bridge.requestCancel()
            val error = failure.asRuntimeError("Generation failed")
            if (stateless) {
                runCatching {
                    withContext(NonCancellable + dispatcher) {
                        operationMutex.withLock {
                            if (loadedModelPath != null) bridge.resetConversation()
                            activeSystemPrompt = null
                        }
                    }
                }
            }
            mutableState.value = RuntimeState.Error(error)
            throw error
        }
    }.flowOn(dispatcher)

    suspend fun generateStructured(
        request: GenerationRequest,
    ): dev.frozenvoice.localai.GenerationResult {
        val output = StringBuilder()
        generate(
            request.copy(
                profile = request.profile.takeUnless { it == GenerationProfile.CHAT }
                    ?: GenerationProfile.STRUCTURED,
                conversationMode = ConversationMode.STATELESS,
            ),
        ).collect { output.append(it) }
        val metrics = lastMetrics() ?: throw LocalAiRuntimeException(
            "Structured generation completed without metrics.",
        )
        return dev.frozenvoice.localai.GenerationResult(output.toString(), metrics)
    }

    suspend fun resetConversation(systemPrompt: String? = null) = withContext(dispatcher) {
        operationMutex.withLock {
            checkReady()
            mutableState.value = RuntimeState.ProcessingPrompt
            try {
                bridge.resetConversation()
                activeSystemPrompt = null
                if (!systemPrompt.isNullOrBlank()) {
                    processSystemPromptLocked(systemPrompt)
                    activeSystemPrompt = systemPrompt
                }
                cancellationRequested.set(false)
                mutableState.value = RuntimeState.ModelReady
            } catch (failure: Throwable) {
                val error = failure.asRuntimeError("Conversation reset failed")
                mutableState.value = RuntimeState.Error(error)
                throw error
            }
        }
    }

    fun cancel() {
        cancellationRequested.set(true)
        runCatching { bridge.requestCancel() }
    }

    suspend fun unload() = withContext(dispatcher) {
        operationMutex.withLock {
            unloadLocked()
            mutableState.value = RuntimeState.Initialized
        }
    }

    fun modelInfo(): ModelInfo? = loadedModel

    fun lastMetrics(): InferenceMetrics? = latestMetrics

    private fun ensureInitialized() {
        if (initialized) return
        mutableState.value = RuntimeState.Initializing
        bridge.initialize(nativeLibraryDir, options)
        initialized = true
        mutableState.value = RuntimeState.Initialized
    }

    private fun checkReady() {
        if (loadedModelPath == null || loadedModel == null) {
            throw LocalAiRuntimeException("A model must be loaded before inference.")
        }
    }

    private fun prepareConversationLocked(
        systemPrompt: String?,
        mode: ConversationMode,
    ) {
        val mustReset = mode != ConversationMode.CONTINUE ||
            (systemPrompt != null && systemPrompt != activeSystemPrompt)
        if (mustReset) {
            bridge.resetConversation()
            activeSystemPrompt = null
        }
        if (systemPrompt != null && systemPrompt != activeSystemPrompt) {
            processSystemPromptLocked(systemPrompt)
            activeSystemPrompt = systemPrompt
        } else if (activeSystemPrompt == null) {
            throw LocalAiRuntimeException(
                "A system prompt is required before the first stateful generation.",
            )
        }
    }

    private fun processSystemPromptLocked(systemPrompt: String) {
        if (bridge.processSystemPrompt(systemPrompt) != 0) {
            throw LocalAiRuntimeException("System prompt processing failed.")
        }
    }

    private fun buildMetrics(request: GenerationRequest): InferenceMetrics {
        val values = bridge.lastMetrics()
        val promptMillis = values.promptProcessingMillis
        val generationMillis = values.generationMillis
        return InferenceMetrics(
            modelLoadMillis = lastLoadMillis,
            timeToFirstTokenMillis = values.timeToFirstTokenMillis,
            totalGenerationMillis = promptMillis + generationMillis,
            promptTokenCount = values.promptTokenCount,
            generatedTokenCount = values.generatedTokenCount,
            promptProcessingMillis = promptMillis,
            generationMillis = generationMillis,
            promptTokensPerSecond = rate(values.promptTokenCount, promptMillis),
            generationTokensPerSecond = rate(values.generatedTokenCount, generationMillis),
            modelReused = lastModelWasReused,
            promptPrefixReused = false,
            runtimeContextLength = loadedRuntimeSettings.contextSize,
            requestedMaxNewTokens = request.maxNewTokens,
            profile = request.profile,
            terminationReason = values.terminationReason,
            runtimeSettings = loadedRuntimeSettings,
        )
    }

    private fun unloadLocked() {
        if (loadedModelPath != null) bridge.unload()
        loadedModelPath = null
        loadedModel = null
        activeSystemPrompt = null
        loadedRuntimeSettings = RuntimeSettings(0, 0, 0, 0, 0, "unknown")
        lastLoadMillis = 0L
        lastModelWasReused = false
        latestMetrics = null
        cancellationRequested.set(true)
    }

    private fun validateModelPath(path: String): String {
        require(path.isNotBlank()) { "Model path must not be blank." }
        val file = File(path).canonicalFile
        require(file.isFile && file.canRead()) {
            "Model path must identify a readable file."
        }
        return file.path
    }

    private fun elapsedMillis(startedAt: Long): Long =
        ((nanoTime() - startedAt).coerceAtLeast(0L) / NANOS_PER_MILLISECOND)

    private companion object {
        const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}

internal data class NativeModelInfo(
    val trainedContextLength: Int?,
    val parameterCount: Long?,
    val hasEmbeddedChatTemplate: Boolean,
    val description: String?,
    val architecture: String?,
)

internal data class NativeMetrics(
    val promptTokenCount: Int,
    val generatedTokenCount: Int,
    val promptProcessingMillis: Long,
    val generationMillis: Long,
    val timeToFirstTokenMillis: Long,
    val terminationReason: TerminationReason,
)

internal interface NativeRuntimeBridge {
    fun initialize(nativeLibraryDir: String, options: RuntimeOptions)
    fun load(modelPath: String)
    fun prepare()
    fun modelInfo(): NativeModelInfo
    fun runtimeSettings(): RuntimeSettings
    fun processSystemPrompt(systemPrompt: String): Int
    fun processUserPrompt(
        prompt: String,
        profile: GenerationProfile,
        maxNewTokens: Int,
        grammar: String?,
    ): Int
    fun generateNextToken(): String?
    fun requestCancel()
    fun resetConversation()
    fun lastMetrics(): NativeMetrics
    fun unload()
}

internal class JniNativeRuntimeBridge : NativeRuntimeBridge {
    override fun initialize(nativeLibraryDir: String, options: RuntimeOptions) {
        withNativeLinkage("Native initialization") {
            System.loadLibrary(LIBRARY_NAME)
            checkNativeResult(
                nativeInitialize(
                    nativeLibraryDir,
                    options.contextSize,
                    options.batchSize,
                    options.threadCount ?: 0,
                    options.batchThreadCount ?: 0,
                ),
                "Native initialization",
            )
        }
    }

    override fun load(modelPath: String) {
        withNativeLinkage("Model loading") {
            checkNativeResult(nativeLoad(modelPath), "Model loading")
        }
    }

    override fun prepare() {
        withNativeLinkage("Runtime preparation") {
            checkNativeResult(nativePrepare(), "Runtime preparation")
        }
    }

    override fun modelInfo(): NativeModelInfo = withNativeLinkage("Model metadata") {
        val values = nativeModelMetadata()
        if (values.size != 4) throw LocalAiRuntimeException("Native model metadata is invalid.")
        NativeModelInfo(
            trainedContextLength = values[0].toInt().takeIf { it > 0 },
            parameterCount = values[1].takeIf { it > 0L },
            hasEmbeddedChatTemplate = values[2] == 1L,
            description = nativeModelDescription().takeIf { it.isNotBlank() },
            architecture = nativeModelArchitecture().takeIf { it.isNotBlank() },
        )
    }

    override fun runtimeSettings(): RuntimeSettings = withNativeLinkage("Runtime settings") {
        val values = nativeRuntimeSettings()
        if (values.size != 6 || values.any { it <= 0 }) {
            throw LocalAiRuntimeException("Native runtime settings are invalid.")
        }
        RuntimeSettings(
            contextSize = values[0].toInt(),
            batchSize = values[1].toInt(),
            microBatchSize = values[2].toInt(),
            threadCount = values[3].toInt(),
            batchThreadCount = values[4].toInt(),
            backend = nativeBackend().ifBlank { "CPU" },
        )
    }

    override fun processSystemPrompt(systemPrompt: String): Int =
        withNativeLinkage("System prompt processing") {
            nativeProcessSystemPrompt(systemPrompt)
        }

    override fun processUserPrompt(
        prompt: String,
        profile: GenerationProfile,
        maxNewTokens: Int,
        grammar: String?,
    ): Int = withNativeLinkage("User prompt processing") {
        nativeProcessUserPrompt(
            prompt,
            maxNewTokens,
            profile.temperature,
            profile.topP,
            profile.topK,
            profile.repeatPenalty,
            if (profile == GenerationProfile.CHAT) -1 else 42,
            grammar,
        )
    }

    override fun generateNextToken(): String? = withNativeLinkage("Token generation") {
        nativeGenerateNextToken()
    }

    override fun requestCancel() {
        runCatching { nativeRequestCancel() }
    }

    override fun resetConversation() {
        withNativeLinkage("Conversation reset") { nativeResetConversation() }
    }

    override fun lastMetrics(): NativeMetrics = withNativeLinkage("Inference metrics") {
        val values = nativeLastInferenceMetrics()
        if (values.size != 12 || values.any { it < 0.0 }) {
            throw LocalAiRuntimeException("Native inference metrics are invalid.")
        }
        NativeMetrics(
            promptTokenCount = values[0].toInt(),
            generatedTokenCount = values[1].toInt(),
            promptProcessingMillis = values[2].toLong(),
            generationMillis = values[3].toLong(),
            timeToFirstTokenMillis = values[4].toLong(),
            terminationReason = terminationReason(values[11].toInt()),
        )
    }

    override fun unload() {
        withNativeLinkage("Native unload") { nativeUnload() }
    }

    private inline fun <T> withNativeLinkage(
        operation: String,
        block: () -> T,
    ): T = try {
        block()
    } catch (failure: LinkageError) {
        throw LocalAiRuntimeException("$operation is unavailable.", failure)
    }

    private fun checkNativeResult(result: Int, operation: String) {
        if (result != 0) throw LocalAiRuntimeException("$operation failed with code $result.")
    }

    private fun terminationReason(code: Int): TerminationReason = when (code) {
        1 -> TerminationReason.EOS
        2 -> TerminationReason.MAX_TOKENS
        3 -> TerminationReason.STOP_SEQUENCE
        4 -> TerminationReason.CANCELLED
        5 -> TerminationReason.ERROR
        else -> TerminationReason.UNKNOWN
    }

    @FastNative
    private external fun nativeInitialize(
        nativeLibraryDir: String,
        contextSize: Int,
        batchSize: Int,
        threadCount: Int,
        batchThreadCount: Int,
    ): Int

    @FastNative private external fun nativeLoad(modelPath: String): Int
    @FastNative private external fun nativePrepare(): Int
    @FastNative private external fun nativeModelMetadata(): LongArray
    @FastNative private external fun nativeModelDescription(): String
    @FastNative private external fun nativeModelArchitecture(): String
    @FastNative private external fun nativeRuntimeSettings(): LongArray
    @FastNative private external fun nativeBackend(): String
    @FastNative private external fun nativeProcessSystemPrompt(systemPrompt: String): Int
    @FastNative
    private external fun nativeProcessUserPrompt(
        userPrompt: String,
        maxNewTokens: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
        seed: Int,
        grammar: String?,
    ): Int

    @FastNative private external fun nativeGenerateNextToken(): String?
    @FastNative private external fun nativeRequestCancel()
    @FastNative private external fun nativeResetConversation()
    @FastNative private external fun nativeLastInferenceMetrics(): DoubleArray
    @FastNative private external fun nativeUnload()

    private companion object {
        const val LIBRARY_NAME = "local-ai-runtime"
    }
}

private fun rate(tokens: Int, millis: Long): Double =
    if (tokens > 0 && millis > 0L) tokens / (millis / 1_000.0) else 0.0

private fun Throwable.asRuntimeError(operation: String): LocalAiRuntimeException =
    if (this is LocalAiRuntimeException) this else LocalAiRuntimeException(
        "$operation: ${message ?: javaClass.simpleName}",
        this,
    )
