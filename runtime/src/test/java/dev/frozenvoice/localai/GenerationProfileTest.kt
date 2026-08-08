package dev.frozenvoice.localai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationProfileTest {
    @Test
    fun structuredProfileUsesDeterministicShortBudget() {
        assertEquals(96, GenerationProfile.STRUCTURED.defaultMaxNewTokens)
        assertEquals(0.0f, GenerationProfile.STRUCTURED.temperature)
        assertEquals(1.0f, GenerationProfile.STRUCTURED.topP)
        assertEquals(40, GenerationProfile.STRUCTURED.topK)
        assertEquals(1.0f, GenerationProfile.STRUCTURED.repeatPenalty)
    }

    @Test
    fun requestRequiresAUsablePrompt() {
        assertThrows(IllegalArgumentException::class.java) {
            GenerationRequest(prompt = "")
        }
        val request = GenerationRequest(prompt = "synthetic", maxNewTokens = 8)
        assertTrue(request.conversationMode == ConversationMode.CONTINUE)
    }
}
