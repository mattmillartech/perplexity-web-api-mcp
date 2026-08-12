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
//
// CHEAPEST WAY TO DO STEP 1 (found 2026-08-12; costs NO query quota, needs no CDP capture):
//   The web picker writes the exact wire value straight into localStorage on selection --
//       pplx:account:<uuid>:pplx.local-user-settings.preferredSearchModels-v1
//         -> {"search":"<the preference string>"}
//   So: open the model picker, click a model, read that key. No query is sent, so no
//   advanced-model quota is consumed (the account hit "No more advanced AI model uses
//   remaining this week" during this work -- quota is a real, small budget).
//   THE THINKING TOGGLE IS THE TRAP. Each thinking-capable row has aria-haspopup="menu" and a
//   SUBMENU containing `menuitemcheckbox "Thinking"` + a switch; the stored preference CHANGES
//   when it is flipped. The picker also renders the toggle state INTO the label, so the same
//   model appears as "Kimi K3" or "Kimi K3 New Thinking" depending on the switch. A label->id
//   pair captured without recording the switch state is one sample of a two-valued function,
//   not a mapping -- that mistake produced a wrong table once already.
//   DO NOT read `...local-user-settings.modelOptionPreferences` for the wire value: its KEY is a
//   family id and its VALUE is the toggle, and the two go inconsistent (it held
//   `grok45low:"reasoning"` while the live preference was `grok45medium`).
//   Reaching the submenu: only Max-gated rows get accessibility refs, so click `menuitemradio`
//   rows by text, then ArrowDown to the CHECKED row (compute its index from the DOM -- fixed
//   counts drift and silently hand you a DIFFERENT model's id) and press ArrowRight.
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
        /// Grok 4.5, thinking OFF. Preference CAPTURED LIVE 2026-08-12 (see the capture note
        /// above): selecting Grok 4.5 with its Thinking switch off wrote `grok45low`, and
        /// flipping the switch changed it to `grok45medium`. NOTE THE SUFFIX IS AN EFFORT
        /// LEVEL, NOT A `_thinking` SUFFIX -- and the ON value is `medium`, NOT `high`, so it
        /// does not follow Gemini's `gemini31pro_high` pattern either. Do not "regularise"
        /// these two strings; they were measured, not derived.
        Grok45 => { name: "grok-4.5", preference: "grok45low" },
        /// Kimi K3, thinking OFF.
        ///
        /// *** THIS PREFERENCE STRING IS NOT VERIFIED. *** Every other entry in this file was
        /// observed on the wire or in the picker's own stored preference; this one was NOT. The
        /// Thinking switch for Kimi K3 read `aria-checked=true` and would not flip in two
        /// attempts, so a non-thinking Kimi id was never observed and may not even exist.
        /// `kimik3` is an operator-sanctioned ASSUMPTION (2026-08-12: "Let's assume that kimi
        /// non-thinking is kimik3"), recorded here explicitly rather than passed off as a
        /// capture, because step 1 above says never guess a preference string.
        /// BEFORE RELYING ON THIS: run the live check in step 4. If Perplexity rejects it,
        /// the fix is to capture the real value, not to widen the parser.
        KimiK3 => { name: "kimi-k3", preference: "kimik3" },
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

    #[test]
    fn grok_45_uses_the_captured_effort_level_preferences() {
        // CAPTURED LIVE 2026-08-12 from the picker's stored preference, both switch states.
        // Pinned as exact literals because they are NOT derivable: the thinking value is an
        // effort level (`medium`), not a `_thinking` suffix, and it is not `high` either -- so
        // neither the Terra/Sonnet convention nor Gemini's `_high` predicts it. If someone
        // "tidies" these into a pattern, this test is what catches it.
        let search: SearchModel = "grok-4.5".parse().expect("search model");
        let reason: ReasonModel = "grok-4.5-thinking".parse().expect("reason model");

        assert_eq!(search.api_preference().as_str(), "grok45low");
        assert_eq!(reason.api_preference().as_str(), "grok45medium");
        assert_ne!(
            search.api_preference().as_str(),
            reason.api_preference().as_str(),
            "thinking must be its own wire value, not a same-model toggle"
        );
    }

    #[test]
    fn kimi_k3_thinking_is_the_captured_value_and_differs_from_the_assumed_base() {
        // `kimik3thinking` WAS observed live. `kimik3` was NOT -- it is an operator-sanctioned
        // assumption (see the doc comment on SearchModel::KimiK3). This test therefore asserts
        // exactly what is known: the captured thinking value, and that the two entries are
        // distinct. It deliberately does NOT claim the base value is correct, because no
        // observation supports that and a passing unit test must not manufacture confidence.
        let reason: ReasonModel = "kimi-k3-thinking".parse().expect("reason model");
        assert_eq!(reason.api_preference().as_str(), "kimik3thinking");

        let search: SearchModel = "kimi-k3".parse().expect("search model");
        assert_ne!(search.api_preference().as_str(), reason.api_preference().as_str());
    }

    #[test]
    fn unknown_model_names_are_rejected_with_the_valid_set() {
        // The parser must refuse an unrecognised name rather than pass it through to the wire;
        // a bad preference string fails at call time, which is the failure mode this file's
        // header warns about.
        let err = "grok-4.5-medium".parse::<SearchModel>().unwrap_err();
        assert!(err.contains("unknown model"), "unexpected error: {err}");
        assert!(err.contains("grok-4.5"), "error should list the valid set: {err}");
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
        /// Grok 4.5 with thinking enabled. Preference CAPTURED LIVE 2026-08-12 by flipping the
        /// Thinking switch and re-reading the picker's stored preference: `grok45low` ->
        /// `grok45medium`. This is a DIFFERENT wire value from the non-thinking entry, which is
        /// the invariant this file already documents for Terra and Sonnet.
        Grok45Thinking => { name: "grok-4.5-thinking", preference: "grok45medium" },
        /// Kimi K3 with thinking enabled. Preference CAPTURED LIVE 2026-08-12 -- this one WAS
        /// observed (the picker stored `kimik3thinking` while the row rendered "Kimi K3 New
        /// Thinking" with its switch on). Contrast the non-thinking `KimiK3` entry above, whose
        /// value is an unverified assumption.
        KimiK3Thinking => { name: "kimi-k3-thinking", preference: "kimik3thinking" },
    }
}
