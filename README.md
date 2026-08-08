# local-ai-runtime

Shared offline Android GGUF inference runtime for Local AI Lab and WONCE.

The `runtime` Android library owns the llama.cpp JNI bridge, GGUF-native metadata,
embedded chat-template formatting, per-request sampler profiles, KV/cache reset,
JSON grammar, cancellation, and inference metrics. Applications provide model
selection and domain prompts; no application business logic is included here.

The repository is intentionally independent of either application. During local
development, sibling checkouts consume it through a Gradle composite build:

```text
E:\Dev\GitHub\local-ai-runtime
E:\Dev\GitHub\kanana-poc
E:\Dev\GitHub\wonce
```

`GenerationRequest` accepts one prompt plus an optional system prompt. Stateful
chat uses repeated `CONTINUE` requests on the same serialized session;
`RESET` starts a new conversation and `STATELESS` clears native KV/chat state
after the request. This keeps the runtime contract explicit instead of
partially interpreting arbitrary message-list snapshots.

The native build is fully disconnected. It contains no GGUF model files; the
runtime has no network client, analytics, or Android `INTERNET` permission.

Build the library with Java 17 and an Android SDK/NDK installation:

```text
gradlew.bat :runtime:test
gradlew.bat :runtime:assembleDebug
```

Third-party license texts are packaged under `runtime/src/main/assets/licenses`.
