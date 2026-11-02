//! The house waiting patterns: how long to keep waiting, and how often to say so.
//!
//! Two ladders, running in OPPOSITE directions, because what waiting means
//! depends on what is being waited for.
//!
//! Waiting on a PERSON, the odds fall with time. Someone who has not answered in
//! ten minutes is probably not at the desk, and the next ten minutes are worth
//! less than the last. So the steps shorten — 10, 5, 3 — and at the end the
//! program decides for itself and carries on. Waiting longer buys less and less.
//!
//! Waiting on a SERVER, the odds do the reverse. A backend still refusing after
//! half an hour is more likely to be in maintenance than to be gone forever, and
//! polling it harder changes nothing. Meanwhile the cost of giving up grows: the
//! work that is lost is everything done so far. So the checkpoints spread out —
//! 5, 10, 15, 20, 30, 60, 90, 120 minutes — and at the end it stops rather than
//! inventing an answer, because there is nothing to invent.
//!
//! Both ladders live here so the two cannot drift apart into unrelated numbers,
//! and so the reason they differ is written down once instead of being rederived
//! every time someone notices they are not the same.

/// Waiting on a person: stage lengths in seconds. 10 minutes, then 5, then 3.
///
/// Shortening on purpose — see the module note. At the end the caller acts on
/// its own (AskUserQuestion takes the recommended option).
pub const ASK_LADDER_SECONDS: [u64; 3] = [600, 300, 180];

/// Waiting on a server: ELAPSED checkpoints in seconds, not stage lengths.
///
/// 5, 10, 15, 20, 30, 60, 90 minutes, ending at 2 hours. The gaps between them
/// widen (5, 5, 5, 5, 10, 30, 30, 30 minutes), which is the point: report often
/// while somebody might still be watching, then back off, because a long outage
/// is not made shorter by asking about it more.
///
/// The last entry is the deadline. Reaching it means calling the outage
/// permanent — a judgement, not a fact, and the one the caller can be wrong
/// about in the cheaper direction: two idle hours on a machine nobody is
/// watching, against losing everything done before it.
pub const RETRY_CHECKPOINTS_SECONDS: [u64; 8] =
    [300, 600, 900, 1200, 1800, 3600, 5400, 7200];

/// Total the ask ladder waits before the caller acts, in whole minutes.
pub fn ask_total_minutes() -> u64 {
    ASK_LADDER_SECONDS.iter().sum::<u64>() / 60
}

/// Minutes left after ask stage `stage` (0-based) has elapsed.
pub fn ask_minutes_left_after(stage: usize) -> u64 {
    ASK_LADDER_SECONDS.iter().skip(stage + 1).sum::<u64>() / 60
}

/// How long the retry ladder keeps trying before giving up, in seconds.
pub fn retry_deadline_secs() -> f64 {
    RETRY_CHECKPOINTS_SECONDS[RETRY_CHECKPOINTS_SECONDS.len() - 1] as f64
}

/// A duration in whole minutes for a message, never rounding down to zero.
///
/// Plain integer division prints "retrying for up to 0 min" for any budget under
/// a minute, which reads as broken rather than as short. Anything non-zero is at
/// least one minute; only a genuinely disabled wait is zero.
pub fn minutes_for_message(secs: f64) -> u64 {
    if secs <= 0.0 {
        return 0;
    }
    ((secs / 60.0).ceil() as u64).max(1)
}

/// How many retry checkpoints `elapsed_secs` has passed.
///
/// The retry loop waits in one continuous stretch rather than in discrete
/// stages, so it asks this after each failure to know whether it has just
/// crossed a line worth reporting.
pub fn retry_checkpoints_passed(elapsed_secs: f64) -> usize {
    RETRY_CHECKPOINTS_SECONDS
        .iter()
        .filter(|cp| elapsed_secs >= **cp as f64)
        .count()
}

/// Minutes remaining until the retry deadline, from `elapsed_secs`.
pub fn retry_minutes_left(elapsed_secs: f64) -> u64 {
    let left = retry_deadline_secs() - elapsed_secs;
    if left <= 0.0 {
        0
    } else {
        (left / 60.0).round() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Waiting on a person shortens; waiting on a server lengthens. If these
    /// ever point the same way, one of them has lost its reason.
    #[test]
    fn the_two_ladders_run_in_opposite_directions() {
        assert!(
            ASK_LADDER_SECONDS.windows(2).all(|w| w[0] > w[1]),
            "the ask ladder must shorten: a person who has not answered is less likely to"
        );
        let gaps: Vec<u64> = RETRY_CHECKPOINTS_SECONDS
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        assert!(
            gaps.windows(2).all(|w| w[1] >= w[0]),
            "retry gaps must never narrow: {gaps:?}"
        );
        assert!(gaps.last() > gaps.first(), "and must actually widen overall");
    }

    /// The numbers the user set, pinned so a later "tidy-up" cannot quietly
    /// change how long this waits.
    #[test]
    fn the_ladders_are_the_agreed_numbers() {
        assert_eq!(ASK_LADDER_SECONDS, [600, 300, 180]);
        assert_eq!(ask_total_minutes(), 18);
        // 5, 10, 15, 20, 30, 60, 90 minutes, ending at 2 hours.
        assert_eq!(RETRY_CHECKPOINTS_SECONDS, [300, 600, 900, 1200, 1800, 3600, 5400, 7200]);
        assert_eq!(retry_deadline_secs(), 7200.0);
    }

    #[test]
    fn ask_progress_counts_down_to_zero() {
        assert_eq!(ask_minutes_left_after(0), 8);
        assert_eq!(ask_minutes_left_after(1), 3);
        assert_eq!(ask_minutes_left_after(2), 0);
    }

    /// A checkpoint is crossed at its exact second, and the count never runs
    /// past the ladder however long the outage lasts.
    #[test]
    fn retry_checkpoints_are_crossed_where_expected() {
        assert_eq!(retry_checkpoints_passed(0.0), 0);
        assert_eq!(retry_checkpoints_passed(299.0), 0);
        assert_eq!(retry_checkpoints_passed(300.0), 1); // 5 min
        assert_eq!(retry_checkpoints_passed(1799.0), 4);
        assert_eq!(retry_checkpoints_passed(1800.0), 5); // 30 min
        assert_eq!(retry_checkpoints_passed(7200.0), 8); // 2 h, the deadline
        assert_eq!(retry_checkpoints_passed(99_999.0), 8, "must not exceed the ladder");
    }

    /// "0 min" for a wait that is actually happening reads as broken.
    #[test]
    fn a_short_wait_is_never_reported_as_zero_minutes() {
        assert_eq!(minutes_for_message(8.0), 1, "8s is short, not nothing");
        assert_eq!(minutes_for_message(59.0), 1);
        assert_eq!(minutes_for_message(60.0), 1);
        assert_eq!(minutes_for_message(7200.0), 120);
        assert_eq!(minutes_for_message(0.0), 0, "only a disabled wait is zero");
    }

    /// The progress line quotes what is left, so it must shrink and bottom out
    /// at zero rather than going negative.
    #[test]
    fn retry_remaining_time_shrinks_and_stops_at_zero() {
        assert_eq!(retry_minutes_left(0.0), 120);
        assert_eq!(retry_minutes_left(300.0), 115);
        assert_eq!(retry_minutes_left(3600.0), 60);
        assert_eq!(retry_minutes_left(7200.0), 0);
        assert_eq!(retry_minutes_left(99_999.0), 0, "never negative");
    }
}
