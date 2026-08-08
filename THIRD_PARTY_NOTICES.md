# local-ai-runtime third-party provenance

This repository contains source code required for offline Android inference. It
does not contain a GGUF model, model download client, or sample application UI.

## llama.cpp

- Project: llama.cpp
- Upstream repository: https://github.com/ggml-org/llama.cpp
- Source snapshot: the validated Local AI Lab nested repository at commit
  `9bb7060e58cb206c35a31befcf6e025d0c8055a7`
- License: MIT
- License text: `runtime/src/main/assets/licenses/llama.cpp-LICENSE.txt`

Only the core CMake, `common`, `cmake`, `ggml`, `include`, `src`, and `vendor`
trees required by the Android runtime are included. Examples, tools, tests, Git
metadata, build outputs, and model files are excluded.

## KleidiAI

- Project: KleidiAI
- Release: `v1.24.0`
- Release archive MD5 pinned by llama.cpp:
  `2f02ebe29573d45813e671eb304f2a00`
- License texts: `runtime/src/main/assets/licenses/KleidiAI-Apache-2.0.txt` and
  `runtime/src/main/assets/licenses/KleidiAI-BSD-3-Clause.txt`

The checked-in source is the already populated, pinned local source tree. CMake
is configured for fully disconnected builds and never downloads this dependency.

## LLVM OpenMP runtime

Arm64 builds enable llama.cpp OpenMP and package the Android NDK `libomp.so`.
The NDK notice is preserved at
`runtime/src/main/assets/licenses/Android-NDK-29.0.13113456-LLVM-NOTICE.txt`.
x86_64 builds disable OpenMP.

## Build pins

- Android NDK: `29.0.13113456`
- CMake: `3.31.6`
- ABIs: `arm64-v8a`, `x86_64`
- Arm64: GGML CPU variants, OpenMP, and KleidiAI enabled
- x86_64: GGML CPU variants enabled; KleidiAI and OpenMP disabled

The runtime suppresses llama.cpp and GGML logs before loading backends or a
model. Prompts, generated output, token pieces, model paths, and GGUF metadata
are not written to Android logs by this library.
