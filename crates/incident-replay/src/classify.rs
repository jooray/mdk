//! The classification gate.
//!
//! Everything downstream (extraction, fault synthesis, replay) is gated behind
//! this verdict, so a healthy export yields zero vectors and a clean exit rather
//! than a crash. Built incrementally, one rule per behaviour.

use std::collections::BTreeMap;
use std::fmt;

use crate::export::{AgentStateExport, EventKind};
use serde::Serialize;

/// How many epochs an engine may trail the group's epoch high-water mark
/// before it counts as left behind. One epoch behind is routine commit
/// propagation; two or more means consecutive commits never arrived.
const EPOCH_DIVERGENCE_MIN_LAG: u64 = 2;

/// How long an engine may keep recording events after the group provably moved
/// past its final epoch before its lag counts as *active while behind* rather
/// than *went dark*. Ordinary catch-up after a reconnect completes in seconds
/// to minutes, and the margin also absorbs cross-device clock skew; the real
/// incidents this gate was validated on stayed behind for six hours (a live
/// device no longer receiving commits) and eighteen hours (exp-07).
const CATCH_UP_GRACE_MS: u64 = 60 * 60 * 1000;

/// How the pipeline should route an export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// No contested branch — the common case. Zero vectors, clean exit.
    Healthy,
    /// A same-epoch commit race resolved by the fork-recovery seam (Phase 3).
    ForkRecovery,
    /// A quiescence-window branch selection (Phase 4; needs the convergence
    /// assert surface).
    ConvergenceSelected,
    /// Unusable for faithful replay; never fabricate a vector from it.
    Quarantine { reason: QuarantineReason },
}

/// Why an export was quarantined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuarantineReason {
    /// A `derived_projections` section was capped server-side (`has_more`), so
    /// the export is incomplete.
    TruncatedProjections,
    /// A fork resolution's winning snapshot was missing — unreproducible.
    MissingSnapshot,
    /// Engines trail the group's epoch high-water mark by at least
    /// [`EPOCH_DIVERGENCE_MIN_LAG`] epochs with no recorded contest explaining
    /// it: the group is silently split. A real liveness incident, but not a
    /// branch contest, so there is nothing to replay as a vector — a human
    /// should look at the named engines instead.
    EpochDivergence {
        /// The group's epoch high-water mark across all engines.
        group_epoch: u64,
        /// Every engine left behind it, in engine-id order.
        engines: Vec<BehindEngine>,
    },
}

impl fmt::Display for QuarantineReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuarantineReason::TruncatedProjections => {
                f.write_str("a derived_projections section was truncated server-side (has_more)")
            }
            QuarantineReason::MissingSnapshot => {
                f.write_str("a fork resolution's winning snapshot was missing")
            }
            QuarantineReason::EpochDivergence {
                group_epoch,
                engines,
            } => {
                write!(f, "engines behind the group tip (epoch {group_epoch}):")?;
                for (index, engine) in engines.iter().enumerate() {
                    let separator = if index == 0 { " " } else { ", " };
                    write!(f, "{separator}{engine}")?;
                }
                Ok(())
            }
        }
    }
}

/// One engine left behind the group's epoch high-water mark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BehindEngine {
    /// The engine that fell behind.
    pub engine_id: String,
    /// The highest epoch the engine's own events place it at.
    pub epoch: u64,
    /// How the engine was behaving once the group moved past it.
    pub mode: BehindMode,
}

impl fmt::Display for BehindEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at epoch {} ({})",
            self.engine_id, self.epoch, self.mode
        )
    }
}

/// How an engine that fell behind was behaving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BehindMode {
    /// The engine stopped recording events before — or within the catch-up
    /// grace of — the group provably advancing past it: a dead device, an
    /// uninstalled app, or stopped uploads. (An engine belonging to a member
    /// who *left* the group looks identical; telling the two apart needs a
    /// member-to-engine linkage the export does not carry yet.)
    WentDark,
    /// The engine kept recording events for longer than the catch-up grace
    /// after the group provably advanced past its final epoch, without
    /// catching up: commits are not reaching it even though its other traffic
    /// flows.
    ActiveWhileBehind,
}

impl fmt::Display for BehindMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BehindMode::WentDark => "went dark",
            BehindMode::ActiveWhileBehind => "active while behind",
        })
    }
}

/// Classify an export into its routing verdict.
pub fn classify(export: &AgentStateExport) -> Verdict {
    // A truncated projection means the export is incomplete: reproduction could
    // miss witnesses or hidden state, so it is unusable regardless of what the
    // (uncapped) event log shows. Gate this first.
    if export
        .derived_projections
        .pagination
        .values()
        .any(|section| section.has_more)
    {
        return Verdict::Quarantine {
            reason: QuarantineReason::TruncatedProjections,
        };
    }
    let kinds = || export.events.iter().map(|event| &event.kind);
    // A contested convergence selection dominates: a real incident can carry
    // both a fork resolution and a convergence decision, and the convergence
    // route (Phase 4) is the one that reproduces it.
    if kinds().any(EventKind::is_contested_convergence) {
        return Verdict::ConvergenceSelected;
    }
    // Below here the export routes to fork recovery, where an unrecoverable
    // winner (missing snapshot) can't be replayed — quarantine instead of
    // fabricating a vector.
    if kinds().any(EventKind::is_missing_snapshot_fork) {
        return Verdict::Quarantine {
            reason: QuarantineReason::MissingSnapshot,
        };
    }
    if kinds().any(EventKind::is_fork_resolution) {
        return Verdict::ForkRecovery;
    }
    // No contested branch anywhere — the liveness gate now guards the healthy
    // verdict. It ranks below the incident routes deliberately: a reproducible
    // contest is worth replaying even when another engine's data is stale
    // (recovery fail-closes downstream if the data it needs is missing). It
    // exists because a stuck or dead device is only visible *across* engines:
    // both real exports this gate was validated on previously classified
    // healthy while genuinely split (2026-07-09 incident: three engines dark
    // and one active engine cut off from commits; exp-07: one engine eighteen
    // hours behind the other).
    if let Some(reason) = epoch_divergence(export) {
        return Verdict::Quarantine { reason };
    }
    Verdict::Healthy
}

/// Per-engine activity, folded from the event log.
#[derive(Default)]
struct EngineActivity {
    /// The engine's newest event timestamp, when its events carry one.
    last_seen_ms: Option<u64>,
    /// The highest epoch the engine reported itself at.
    high_water_epoch: Option<u64>,
}

/// The gate that separates "no contested branch" from "healthy": engines left
/// ≥ [`EPOCH_DIVERGENCE_MIN_LAG`] epochs behind the group's high-water mark.
///
/// It fires only on positive evidence — events without an `engine_id` or
/// `wall_time_ms` leave it unarmed — so synthetic fixtures and older exports
/// classify as before. Lag is measured in epochs, not wall-clock silence: a
/// device that is merely offline while nothing is committed misses nothing and
/// stays healthy, and an idle group never reads as stale.
fn epoch_divergence(export: &AgentStateExport) -> Option<QuarantineReason> {
    let mut engines: BTreeMap<&str, EngineActivity> = BTreeMap::new();
    // Per epoch, the earliest timed evidence of it from any engine: the moment
    // after which staying behind that epoch stops being propagation delay.
    let mut epoch_first_seen: BTreeMap<u64, u64> = BTreeMap::new();
    for event in &export.events {
        let Some(engine_id) = event.engine_id.as_deref() else {
            continue;
        };
        let activity = engines.entry(engine_id).or_default();
        activity.last_seen_ms = activity.last_seen_ms.max(event.wall_time_ms);
        let observed = event.kind.observed_epoch();
        activity.high_water_epoch = activity.high_water_epoch.max(observed);
        if let (Some(epoch), Some(ms)) = (observed, event.wall_time_ms) {
            epoch_first_seen
                .entry(epoch)
                .and_modify(|first| *first = (*first).min(ms))
                .or_insert(ms);
        }
    }

    let group_epoch = engines
        .values()
        .filter_map(|activity| activity.high_water_epoch)
        .max()?;
    let behind: Vec<BehindEngine> = engines
        .iter()
        .filter_map(|(engine_id, activity)| {
            let epoch = activity.high_water_epoch?;
            if group_epoch - epoch < EPOCH_DIVERGENCE_MIN_LAG {
                return None;
            }
            // Both the engine's own liveness and the group's advance past it
            // must be timestamped to order them; untimed evidence stays
            // unarmed rather than guessing.
            let last_seen = activity.last_seen_ms?;
            let moved_past = epoch_first_seen
                .range(epoch + 1..)
                .map(|(_, first_seen)| *first_seen)
                .min()?;
            let mode = if last_seen > moved_past + CATCH_UP_GRACE_MS {
                BehindMode::ActiveWhileBehind
            } else {
                BehindMode::WentDark
            };
            Some(BehindEngine {
                engine_id: (*engine_id).to_owned(),
                epoch,
                mode,
            })
        })
        .collect();

    (!behind.is_empty()).then_some(QuarantineReason::EpochDivergence {
        group_epoch,
        engines: behind,
    })
}
