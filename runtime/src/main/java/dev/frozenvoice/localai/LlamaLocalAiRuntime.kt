package dev.frozenvoice.localai

import android.content.Context
import dev.frozenvoice.localai.internal.InferenceEngineImpl

/** Default llama.cpp-backed implementation of [LocalAiRuntime]. */
class LlamaLocalAiRuntime(
    context: Context,
    options: RuntimeOptions = RuntimeOptions(),
) : LocalAiRuntime by InferenceEngineImpl(context.applicationContext ?: context, options)

object LocalAiRuntimeFactory {
    fun create(
        context: Context,
        options: RuntimeOptions = RuntimeOptions(),
    ): LocalAiRuntime = LlamaLocalAiRuntime(context, options)
}
