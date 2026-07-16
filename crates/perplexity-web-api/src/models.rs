use std::fmt;
use std::str::FromStr;

pub const DEEP_RESEARCH_MODEL_PREFERENCE: &str = "pplx_alpha";

/// A validated model preference string sent to the Perplexity API payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPreference(&'static str);

impl ModelPreference {
    /// Returns the raw API model preference value.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

macro_rules! define_model_enum {
    (
        $(#[$enum_meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => { name: $model_name:literal, preference: $preference:literal }
            ),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub const VALID_NAMES: &'static [&'static str] = &[$($model_name),+];

            pub const fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $model_name,)+
                }
            }

            pub const fn api_preference(&self) -> ModelPreference {
                match self {
                    $(Self::$variant => ModelPreference($preference),)+
                }
            }

            pub fn valid_names_csv() -> String {
                Self::VALID_NAMES.join(", ")
            }
        }

        impl From<$name> for ModelPreference {
            fn from(value: $name) -> Self {
                value.api_preference()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($model_name => Ok(Self::$variant),)+
                    _ => Err(format!(
                        "unknown model '{s}', expected one of: {}",
                        Self::valid_names_csv()
                    )),
                }
            }
        }

        impl TryFrom<&str> for $name {
            type Error = String;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                value.parse()
            }
        }
    };
}

// ADDING A NEW MODEL (read this first -- 2026-07-16 incident):
//   1. Discover the real name/preference pair via a live network capture against an
//      authenticated browser session (see ../../../../vault/brain/gotchas/
//      perplexity-mcp-space-fork.md, "CDP capture") -- never guess a preference string.
//   2. Add the entry here (and to ReasonModel below if a thinking variant exists -- it is
//      its OWN model entry with its OWN preference string, e.g. "gpt56_terra_thinking" is
//      NOT the same wire value as "gpt56_terra" -- thinking is not a same-model toggle).
//   3. The #[test] below only proves this Rust code maps the name to the preference string
//      correctly -- it does NOT prove Perplexity's backend actually accepts that preference.
//      Those are different claims; conflating them cost real time once already (see
//      ../../../../vault/brain/decisions/2026-07-16-perplexity-model-preference-verification.md).
//   4. Add the same name to ASK_MODELS/REASON_MODELS in vault's skills/perplexity/perplexity.py,
//      then run `python3 skills/perplexity/verify_live_models.py` (real end-to-end call against
//      the live account) to prove step 3's distinction actually holds for the new model.
define_model_enum! {
    /// Model selection for `perplexity_search`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SearchModel {
        /// Default (auto) free model
        Turbo => { name: "turbo", preference: "turbo" },
        /// Pro auto (best) model.
        ProAuto => { name: "pro-auto", preference: "pplx_pro" },
        /// Sonar model.
        Sonar => { name: "sonar", preference: "experimental" },
        /// GPT-5.4 model.
        Gpt54 => { name: "gpt-5.4", preference: "gpt54" },
        /// GPT-5.6 Terra model. Live end-to-end verified 2026-07-16 (not just this file's
        /// unit test) -- see verify_live_models.py / the decision doc referenced above.
        Gpt56Terra => { name: "gpt-5.6-terra", preference: "gpt56_terra" },
        /// Claude Sonnet 5.0 model. Live end-to-end verified 2026-07-16, same basis as Terra.
        Claude50Sonnet => { name: "claude-5.0-sonnet", preference: "claude50sonnet" },
        /// Nemotron 3 Super
        Nemotron3Super => { name: "nemotron-3-super", preference: "nv_nemotron_3_super" },
    }
}

#[cfg(test)]
mod tests {
    use super::{ReasonModel, SearchModel};

    // These assert STRING MAPPING correctness only (name -> preference), not that Perplexity's
    // backend accepts the preference or returns a real answer -- that is a separate, stronger
    // claim, only proven by an actual live call. See verify_live_models.py in vault's
    // skills/perplexity/ for the real end-to-end check (run manually, not part of `cargo test`
    // since it hits the live authenticated account); see brain/decisions/2026-07-16-perplexity-
    // model-preference-verification.md for the full methodology + results this was verified
    // against as of 2026-07-16.
    #[test]
    fn gpt_56_terra_uses_the_live_perplexity_preference() {
        let search: SearchModel = "gpt-5.6-terra".parse().expect("search model");
        let reason: ReasonModel = "gpt-5.6-terra-thinking".parse().expect("reason model");

        assert_eq!(search.api_preference().as_str(), "gpt56_terra");
        assert_eq!(reason.api_preference().as_str(), "gpt56_terra_thinking");
    }

    #[test]
    fn claude_sonnet_5_thinking_is_a_distinct_preference_from_non_thinking() {
        // Live end-to-end verified 2026-07-16: thinking is its own model entry with its own
        // wire preference, not a same-model runtime toggle. True for BOTH Sonnet and Terra,
        // confirmed independently for each -- do not assume one implies the other for a
        // future model; each needs its own live check via verify_live_models.py.
        let ask: SearchModel = "claude-5.0-sonnet".parse().expect("search model");
        let reason: ReasonModel = "claude-5.0-sonnet-thinking".parse().expect("reason model");
        assert_ne!(ask.api_preference().as_str(), reason.api_preference().as_str());
    }
}

define_model_enum! {
    /// Model selection for `perplexity_reason`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ReasonModel {
        /// Gemini 3.1 Pro model.
        Gemini31Pro => { name: "gemini-3.1-pro", preference: "gemini31pro_high" },
        /// GPT-5.4 with thinking capabilities.
        Gpt54Thinking => { name: "gpt-5.4-thinking", preference: "gpt54_thinking" },
        /// GPT-5.6 Terra with thinking capabilities. Live end-to-end verified 2026-07-16 --
        /// see verify_live_models.py / the decision doc referenced above the SearchModel enum.
        Gpt56TerraThinking => { name: "gpt-5.6-terra-thinking", preference: "gpt56_terra_thinking" },
        /// Claude Sonnet 5.0 with thinking enabled. Live end-to-end verified 2026-07-16, same
        /// basis as Terra Thinking.
        Claude50SonnetThinking => { name: "claude-5.0-sonnet-thinking", preference: "claude50sonnetthinking" },
    }
}
