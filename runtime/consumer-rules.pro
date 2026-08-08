# The public API is consumed by app code; R8 can otherwise remove JNI entry points.
-keep class dev.frozenvoice.localai.internal.InferenceEngineImpl { *; }
