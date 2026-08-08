#include <android/log.h>
#include <jni.h>
#include <atomic>
#include <string>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 8192;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.7f;
constexpr float DEFAULT_SAMPLER_TOP_P   = 0.9f;
constexpr int   DEFAULT_SAMPLER_TOP_K   = 40;
constexpr float DEFAULT_REPEAT_PENALTY  = 1.08f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
static int                                 g_context_size = DEFAULT_CONTEXT_SIZE;
static int                                 g_batch_size = BATCH_SIZE;
static int                                 g_thread_count = 0;
static int                                 g_batch_thread_count = 0;
static bool                                g_backend_initialized = false;
static bool                               g_has_embedded_chat_template;
static bool                               g_use_jinja_template;
static std::string                        g_chat_template_source;
static std::atomic_bool                   g_cancel_requested { false };

static int                                 g_last_prompt_tokens;
static int                                 g_last_generated_tokens;
static int64_t                             g_last_prompt_processing_us;
static int64_t                             g_generation_started_us;
static int64_t                             g_last_generation_us;
static int64_t                             g_time_to_first_token_us;
static bool                                g_generation_active;
static int                                 g_last_requested_max_new_tokens;
static float                               g_last_temperature;
static float                               g_last_top_p;
static int                                 g_last_top_k;
static float                               g_last_repeat_penalty;
static std::string                          g_last_termination_reason = "UNKNOWN";

static void reset_inference_metrics() {
    g_last_prompt_tokens = 0;
    g_last_generated_tokens = 0;
    g_last_prompt_processing_us = 0;
    g_generation_started_us = 0;
    g_last_generation_us = 0;
    g_time_to_first_token_us = 0;
    g_generation_active = false;
    g_last_requested_max_new_tokens = 0;
    g_last_temperature = 0.0f;
    g_last_top_p = 0.0f;
    g_last_top_k = 0;
    g_last_repeat_penalty = 0.0f;
    g_last_termination_reason = "UNKNOWN";
}

static void update_generation_elapsed() {
    if (g_generation_active) g_last_generation_us = ggml_time_us() - g_generation_started_us;
}

static void set_termination_reason(const char * reason) {
    g_last_termination_reason = reason ? reason : "UNKNOWN";
}

static int termination_reason_code() {
    if (g_last_termination_reason == "EOS") return 1;
    if (g_last_termination_reason == "MAX_TOKENS") return 2;
    if (g_last_termination_reason == "STOP_SEQUENCE") return 3;
    if (g_last_termination_reason == "CANCELLED") return 4;
    if (g_last_termination_reason == "ERROR") return 5;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeInitialize(
        JNIEnv *env,
        jobject /*unused*/,
        jstring nativeLibDir,
        jint context_size,
        jint batch_size,
        jint thread_count,
        jint batch_thread_count) {
    g_context_size = std::max(256, static_cast<int>(context_size));
    g_batch_size = std::max(1, std::min(static_cast<int>(batch_size), g_context_size));
    g_thread_count = thread_count > 0 ? thread_count : 0;
    g_batch_thread_count = batch_thread_count > 0 ? batch_thread_count : 0;

    llama_log_set(aichat_android_log_callback, nullptr);
    if (g_backend_initialized) {
        return 0;
    }

    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    if (!path_to_backend || path_to_backend[0] == '\0') {
        if (path_to_backend) env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);
        return 1;
    }
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);
    llama_backend_init();
    g_backend_initialized = true;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeLoad(
        JNIEnv *env,
        jobject,
        jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGd("%s: Loading model from: \n%s\n", __func__, model_path);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    return 0;
}

static llama_context *init_context(llama_model *model, const int n_ctx = -1) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    const int detected_threads = std::max(
            N_THREADS_MIN,
            std::min(
                    N_THREADS_MAX,
                    static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)) - N_THREADS_HEADROOM));
    const int n_threads = g_thread_count > 0 ? g_thread_count : detected_threads;
    const int n_threads_batch =
            g_batch_thread_count > 0 ? g_batch_thread_count : n_threads;
    const int requested_context = n_ctx > 0 ? n_ctx : g_context_size;
    if (n_ctx <= 0) {
        g_thread_count = n_threads;
        g_batch_thread_count = n_threads_batch;
    }

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (requested_context > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, requested_context);
    }
    ctx_params.n_ctx = requested_context;
    ctx_params.n_batch = g_batch_size;
    ctx_params.n_ubatch = g_batch_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads_batch;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler(
        const float temperature,
        const float top_p,
        const int top_k,
        const float repeat_penalty,
        const int seed,
        const std::string & grammar) {
    common_params_sampling sparams;
    sparams.temp = temperature;
    sparams.top_p = top_p;
    sparams.top_k = top_k;
    sparams.min_p = 0.0f;
    sparams.penalty_repeat = repeat_penalty;
    if (seed >= 0) {
        sparams.seed = static_cast<uint32_t>(seed);
    }
    if (!grammar.empty()) {
        sparams.grammar = { COMMON_GRAMMAR_TYPE_USER, grammar };
    }

    LOGi(
        "sampler_config: temperature=%.3f top_p=%.3f top_k=%d repeat_penalty=%.3f "
        "seed=%u grammar=%s stop_sequences=[]",
        sparams.temp,
        sparams.top_p,
        sparams.top_k,
        sparams.penalty_repeat,
        sparams.seed,
        grammar.empty() ? "none" : "json");
    return common_sampler_init(g_model, sparams);
}

static bool configure_sampler(
        const float temperature,
        const float top_p,
        const int top_k,
        const float repeat_penalty,
        const int seed,
        const std::string & grammar) {
    common_sampler * next = nullptr;
    try {
        next = new_sampler(temperature, top_p, top_k, repeat_penalty, seed, grammar);
    } catch (const std::exception & error) {
        LOGe("sampler_config failed: %s", error.what());
        set_termination_reason("ERROR");
        return false;
    }
    if (!next) {
        LOGe("sampler_config failed: common_sampler_init returned null");
        set_termination_reason("ERROR");
        return false;
    }
    if (g_sampler) {
        common_sampler_free(g_sampler);
    }
    g_sampler = next;
    return true;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativePrepare(
        JNIEnv * /*env*/,
        jobject /*unused*/) {
    auto *context = init_context(g_model);
    if (!context) { return 1; }
    g_context = context;
    g_batch = llama_batch_init(g_batch_size, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    const char * embedded_template = llama_model_chat_template(g_model, nullptr);
    g_has_embedded_chat_template = embedded_template != nullptr && embedded_template[0] != '\0';
    g_use_jinja_template = g_has_embedded_chat_template;
    g_chat_template_source = common_chat_templates_source(g_chat_templates.get());
    LOGi(
        "chat_template: embedded=%s explicit=%s use_jinja=%s source:\n%s",
        g_has_embedded_chat_template ? "yes" : "no",
        common_chat_templates_was_explicit(g_chat_templates.get()) ? "yes" : "no",
        g_use_jinja_template ? "yes" : "no",
        g_chat_template_source.c_str());
    g_sampler = new_sampler(
        DEFAULT_SAMPLER_TEMP,
        DEFAULT_SAMPLER_TOP_P,
        DEFAULT_SAMPLER_TOP_K,
        DEFAULT_REPEAT_PENALTY,
        -1,
        "");
    reset_inference_metrics();
    return 0;
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeModelMetadata(
        JNIEnv *env,
        jobject /*unused*/) {
    if (g_model == nullptr) return nullptr;
    const jlong values[] = {
            static_cast<jlong>(llama_model_n_ctx_train(g_model)),
            static_cast<jlong>(llama_model_n_params(g_model)),
            llama_model_chat_template(g_model, nullptr) != nullptr ? 1 : 0,
            static_cast<jlong>(llama_model_size(g_model)),
    };
    auto result = env->NewLongArray(4);
    if (result != nullptr) env->SetLongArrayRegion(result, 0, 4, values);
    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeModelDescription(
        JNIEnv *env,
        jobject /*unused*/) {
    if (g_model == nullptr) return env->NewStringUTF("");
    char description[512] = {};
    const int length = llama_model_desc(g_model, description, sizeof(description));
    return env->NewStringUTF(length > 0 ? description : "");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeModelArchitecture(
        JNIEnv *env,
        jobject /*unused*/) {
    if (g_model == nullptr) return env->NewStringUTF("");
    char architecture[128] = {};
    const int length = llama_model_meta_val_str(
            g_model,
            "general.architecture",
            architecture,
            sizeof(architecture));
    return env->NewStringUTF(length > 0 ? architecture : "");
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeRuntimeSettings(
        JNIEnv *env,
        jobject /*unused*/) {
    if (g_context == nullptr || g_thread_count <= 0 || g_batch_thread_count <= 0) {
        return nullptr;
    }
    const jlong values[] = {
            static_cast<jlong>(llama_n_ctx(g_context)),
            static_cast<jlong>(g_batch_size),
            static_cast<jlong>(g_batch_size),
            static_cast<jlong>(g_thread_count),
            static_cast<jlong>(g_batch_thread_count),
            1,
    };
    auto result = env->NewLongArray(6);
    if (result != nullptr) env->SetLongArrayRegion(result, 0, 6, values);
    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeBackend(
        JNIEnv *env,
        jobject /*unused*/) {
    return env->NewStringUTF("CPU");
}

/*
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
*/

/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * Context shifting discards the older half of the tokens appended after the system prompt.
 * A future native implementation may replace this with a model-specific sliding window:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the latest tokens appended after the system prompt
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static void log_chat_history() {
    for (size_t i = 0; i < chat_msgs.size(); ++i) {
        const auto & message = chat_msgs[i];
        LOGi("final_message[%zu] role=%s content:\n%s", i, message.role.c_str(), message.content.c_str());
    }
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    std::string formatted;
    try {
        formatted = common_chat_format_single(
                g_chat_templates.get(),
                chat_msgs,
                new_msg,
                role == ROLE_USER,
                g_use_jinja_template);
    } catch (const std::exception & error) {
        if (!g_use_jinja_template) {
            throw;
        }
        // Kanana's embedded template contains tool/multistep branches that
        // require a user query during Jinja parser generation. The Android
        // sample formats one message at a time, so use llama.cpp's legacy
        // embedded-template C API for this incremental path.
        LOGw(
            "chat_template: Jinja formatting failed for role=%s: %s; "
            "falling back to embedded legacy formatter",
            role.c_str(),
            error.what());
        g_use_jinja_template = false;
        formatted = common_chat_format_single(
                g_chat_templates.get(),
                chat_msgs,
                new_msg,
                role == ROLE_USER,
                false);
    }
    chat_msgs.push_back(new_msg);
    LOGi(
        "%s: role=%s use_jinja=%s history_size=%zu formatted_delta:\n%s",
        __func__,
        role.c_str(),
        g_use_jinja_template ? "yes" : "no",
        chat_msgs.size(),
        formatted.c_str());
    log_chat_history();
    return formatted;
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

extern "C"
JNIEXPORT void JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeRequestCancel(
        JNIEnv * /*unused*/,
        jobject /*unused*/
) {
    g_cancel_requested.store(true);
    if (g_generation_active) {
        update_generation_elapsed();
        g_generation_active = false;
    }
    set_termination_reason("CANCELLED");
    LOGi("%s: cancellation requested", __func__);
}
extern "C"
JNIEXPORT void JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeResetConversation(
        JNIEnv * /*unused*/,
        jobject /*unused*/
) {
    reset_long_term_states();
    reset_short_term_states();
    g_cancel_requested.store(false);
    if (g_sampler) common_sampler_reset(g_sampler);
}

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeLastInferenceMetrics(
        JNIEnv *env,
        jobject /*unused*/
) {
    update_generation_elapsed();
    const jdouble values[] = {
        static_cast<jdouble>(g_last_prompt_tokens),
        static_cast<jdouble>(g_last_generated_tokens),
        static_cast<jdouble>(g_last_prompt_processing_us) / 1000.0,
        static_cast<jdouble>(g_last_generation_us) / 1000.0,
        static_cast<jdouble>(g_time_to_first_token_us) / 1000.0,
        g_context ? static_cast<jdouble>(llama_n_ctx(g_context)) : 0.0,
        static_cast<jdouble>(g_last_requested_max_new_tokens),
        static_cast<jdouble>(g_last_temperature),
        static_cast<jdouble>(g_last_top_p),
        static_cast<jdouble>(g_last_top_k),
        static_cast<jdouble>(g_last_repeat_penalty),
        static_cast<jdouble>(termination_reason_code()),
    };
    auto result = env->NewDoubleArray(12);
    if (result != nullptr) env->SetDoubleArrayRegion(result, 0, 12, values);
    return result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeRuntimeContextLength(
        JNIEnv * /*unused*/,
        jobject /*unused*/
) {
    return g_context ? static_cast<jint>(llama_n_ctx(g_context)) : 0;
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // Process tokens in batches using the global batch
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += g_batch_size) {
        const int cur_batch_size = std::min((int) tokens.size() - i, g_batch_size);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= g_context_size - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeProcessSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = g_has_embedded_chat_template;
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = g_context_size - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeProcessUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict,
        jfloat temperature,
        jfloat top_p,
        jint top_k,
        jfloat repeat_penalty,
        jint seed,
        jstring jgrammar
) {
    reset_short_term_states();
    reset_inference_metrics();
    g_cancel_requested.store(false);

    g_last_requested_max_new_tokens = std::max(1, static_cast<int>(n_predict));
    g_last_temperature = temperature;
    g_last_top_p = top_p;
    g_last_top_k = top_k;
    g_last_repeat_penalty = repeat_penalty;

    std::string grammar;
    if (jgrammar != nullptr) {
        const auto * grammar_chars = env->GetStringUTFChars(jgrammar, nullptr);
        if (grammar_chars != nullptr) {
            grammar = grammar_chars;
            env->ReleaseStringUTFChars(jgrammar, grammar_chars);
        }
    }
    LOGi("%s: JNI config max_new_tokens=%d temperature=%.3f top_p=%.3f top_k=%d repeat_penalty=%.3f grammar=%s",
        __func__, g_last_requested_max_new_tokens, g_last_temperature, g_last_top_p,
        g_last_top_k, g_last_repeat_penalty, grammar.empty() ? "none" : "json");

    if (!configure_sampler(g_last_temperature, g_last_top_p, g_last_top_k,
                           g_last_repeat_penalty, seed, grammar)) {
        return 3;
    }

    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received:\n%s", __func__, user_prompt);
    std::string formatted_user_prompt(user_prompt);
    const bool has_chat_template = g_has_embedded_chat_template;
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    LOGi("final_prompt: role_history=%zu template_embedded=%s use_jinja=%s formatted_user_delta:\n%s",
        chat_msgs.size(), has_chat_template ? "yes" : "no",
        g_use_jinja_template ? "yes" : "no", formatted_user_prompt.c_str());

    auto user_tokens = common_tokenize(g_context, formatted_user_prompt,
                                       has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: %s -> %d", common_token_to_piece(g_context, id).c_str(), id);
    }

    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = g_context_size - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    const int effective_user_prompt_size = (int) user_tokens.size();
    const int64_t prompt_processing_started_us = ggml_time_us();
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        set_termination_reason("ERROR");
        return 2;
    }
    g_last_prompt_processing_us = ggml_time_us() - prompt_processing_started_us;
    g_last_prompt_tokens = effective_user_prompt_size;
    current_position += effective_user_prompt_size;
    stop_generation_position = current_position + g_last_requested_max_new_tokens;
    g_generation_started_us = ggml_time_us();
    g_generation_active = true;
    LOGi("generation_start: position=%d stop_position=%d max_new_tokens=%d stop_sequences=[]",
        current_position, stop_generation_position, g_last_requested_max_new_tokens);
    return 0;
}
static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeGenerateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Infinite text generation via context shifting
    if (g_cancel_requested.load()) {
        update_generation_elapsed();
        g_generation_active = false;
        set_termination_reason("CANCELLED");
        LOGi("%s: CANCELLED by caller", __func__);
        return nullptr;
    }

    if (g_last_generated_tokens >= g_last_requested_max_new_tokens) {
        update_generation_elapsed();
        g_generation_active = false;
        set_termination_reason("MAX_TOKENS");
        return nullptr;
    }

    if (current_position >= g_context_size - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);
    if (g_time_to_first_token_us == 0) {
        g_time_to_first_token_us = ggml_time_us() - g_generation_started_us;
    }

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        update_generation_elapsed();
        g_generation_active = false;
        set_termination_reason("ERROR");
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        update_generation_elapsed();
        g_generation_active = false;
        set_termination_reason("EOS");
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        LOGi(
            "generation_end: reason=EOS generated_tokens=%d output:\n%s",
            g_last_generated_tokens,
            assistant_ss.str().c_str());
        return nullptr;
    }

    g_last_generated_tokens++;
    update_generation_elapsed();
    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_dev_frozenvoice_localai_internal_JniNativeRuntimeBridge_nativeUnload(
        JNIEnv * /*unused*/,
        jobject /*unused*/) {
    // Reset long-term & short-term states
    if (g_context != nullptr) reset_long_term_states();
    reset_short_term_states();
    g_cancel_requested.store(true);
    reset_inference_metrics();

    // Free up resources
    if (g_sampler != nullptr) common_sampler_free(g_sampler);
    g_chat_templates.reset();
    if (g_batch.token != nullptr) llama_batch_free(g_batch);
    if (g_context != nullptr) llama_free(g_context);
    if (g_model != nullptr) llama_model_free(g_model);
    g_sampler = nullptr;
    g_chat_template_source.clear();
    g_has_embedded_chat_template = false;
    g_use_jinja_template = false;
    g_context = nullptr;
    g_model = nullptr;
    g_chat_templates.reset();
    g_batch = {};
}
