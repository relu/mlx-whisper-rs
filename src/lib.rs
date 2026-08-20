pub mod audio;
pub mod decoding;
pub mod load_models;
pub mod tokenizer;
pub mod transcribe;
pub mod whisper;

/// The golden-fixture helpers, shared with the integration tests.
///
/// `tests/common/mod.rs` is pulled in here rather than copied, so the fixture
/// path, the assets-missing skip, the `MLX_WHISPER_RS_TEST_CPU` device pin and
/// the synthetic-logit LCG have exactly one definition. The in-crate tests need
/// them because the logit filters are private and their tests must live beside
/// them; the integration tests reach the same file through `mod common;`.
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_common;
