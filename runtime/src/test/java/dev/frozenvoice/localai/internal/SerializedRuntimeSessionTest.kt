package dev.frozenvoice.localai.internal

import dev.frozenvoice.localai.ConversationMode
import dev.frozenvoice.localai.GenerationProfile
import dev.frozenvoice.localai.GenerationRequest
import dev.frozenvoice.localai.RuntimeOptions
import dev.frozenvoice.localai.RuntimeSettings
import dev.frozenvoice.localai.TerminationReason
import java.nio.file.Files
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class SerializedRuntimeSessionTest {
    @Test
    fun statelessGenerationResetsConversationAndAppliesRequestSampler() = runTest {
        val model = Files.createTempFile("local-ai-runtime-test-", ".gguf").toFile()
        try {
            val bridge = RecordingBridge()
            val session = SerializedRuntimeSession(
                nativeLibraryDir = "/synthetic/lib",
                options = RuntimeOptions(contextSize = 2_048),
                bridge = bridge,
                dispatcher = UnconfinedTestDispatcher(testScheduler),
            )

            session.loadModel(model.absolutePath)
            val secondInfo = session.loadModel(model.absolutePath)
            val output = session.generate(
                GenerationRequest(
                    prompt = "synthetic request",
                    systemPrompt = "synthetic policy",
                    profile = GenerationProfile.STRUCTURED,
                    maxNewTokens = 12,
                    grammar = "root ::= \"ok\"",
                    conversationMode = ConversationMode.STATELESS,
                ),
            ).toList().joinToString("")

            assertEquals(1, bridge.loadCount)
            assertEquals(0L, secondInfo.loadTimeMillis)
            assertEquals("AB", output)
            assertEquals(GenerationProfile.STRUCTURED, bridge.lastProfile)
            assertEquals(12, bridge.lastMaxNewTokens)
            assertEquals("root ::= \"ok\"", bridge.lastGrammar)
            assertTrue(bridge.resetCount >= 2)
            assertEquals(TerminationReason.EOS, session.lastMetrics()?.terminationReason)
        } finally {
            model.delete()
        }
    }

    @Test
    fun statelessGenerationResetsConversationAfterNativeFailure() = runTest {
        val model = Files.createTempFile("local-ai-runtime-error-test-", ".gguf").toFile()
        try {
            val bridge = RecordingBridge(failPrompt = true)
            val session = SerializedRuntimeSession(
                nativeLibraryDir = "/synthetic/lib",
                options = RuntimeOptions(contextSize = 2_048),
                bridge = bridge,
                dispatcher = UnconfinedTestDispatcher(testScheduler),
            )

            session.loadModel(model.absolutePath)
            val failure = runCatching {
                session.generate(
                    GenerationRequest(
                        prompt = "synthetic failure",
                        systemPrompt = "synthetic policy",
                        conversationMode = ConversationMode.STATELESS,
                    ),
                ).toList()
            }.exceptionOrNull()

            assertTrue(failure is RuntimeException)
            assertTrue(bridge.resetCount >= 2)
        } finally {
            model.delete()
        }
    }

    private class RecordingBridge(
        private val failPrompt: Boolean = false,
    ) : NativeRuntimeBridge {
        var loadCount = 0
        var resetCount = 0
        var lastProfile: GenerationProfile? = null
        var lastMaxNewTokens: Int? = null
        var lastGrammar: String? = null
        private var nextTokenIndex = 0

        override fun initialize(nativeLibraryDir: String, options: RuntimeOptions) = Unit

        override fun load(modelPath: String) {
            loadCount += 1
        }

        override fun prepare() = Unit

        override fun modelInfo() = NativeModelInfo(
            trainedContextLength = 2_048,
            parameterCount = 3L,
            hasEmbeddedChatTemplate = true,
            description = "synthetic model",
            architecture = "synthetic",
        )

        override fun runtimeSettings() = RuntimeSettings(2_048, 512, 512, 2, 2, "CPU")

        override fun processSystemPrompt(systemPrompt: String): Int = 0

        override fun processUserPrompt(
            prompt: String,
            profile: GenerationProfile,
            maxNewTokens: Int,
            grammar: String?,
        ): Int {
            lastProfile = profile
            lastMaxNewTokens = maxNewTokens
            lastGrammar = grammar
            nextTokenIndex = 0
            return if (failPrompt) 1 else 0
        }

        override fun generateNextToken(): String? = when (nextTokenIndex++) {
            0 -> "A"
            1 -> "B"
            else -> null
        }

        override fun requestCancel() = Unit

        override fun resetConversation() {
            resetCount += 1
        }

        override fun lastMetrics() = NativeMetrics(
            promptTokenCount = 4,
            generatedTokenCount = 2,
            promptProcessingMillis = 10L,
            generationMillis = 20L,
            timeToFirstTokenMillis = 5L,
            terminationReason = TerminationReason.EOS,
        )

        override fun unload() = Unit
    }
}
