#pragma once

#include "ggml.h"

// Inference requests can contain financial or otherwise private user text. The
// shared runtime deliberately keeps llama.cpp and application logs silent.
#define LOGv(...) ((void)0)
#define LOGd(...) ((void)0)
#define LOGi(...) ((void)0)
#define LOGw(...) ((void)0)
#define LOGe(...) ((void)0)

static inline void aichat_android_log_callback(
        enum ggml_log_level,
        const char *,
        void *) {}
