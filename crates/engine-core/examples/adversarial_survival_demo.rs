//! Reproducible adversarial survival experiment for online model adaptation.
//!
//! The opponent changes its counter-strategy on a hidden schedule. The agent
//! observes only its own survival delta, not the opponent's phase. There is no
//! action, energy, or compute budget; survival ends only when health reaches zero.
//! The experiment compares a cumulative NumericDeltaModel with a recent-window
//! variant that can discard stale evidence after opponent strategy changes.

use engine_core::{
    AdaptiveLearningPolicy, AdaptiveNumericDeltaModel, DataValue, NumericDeltaModel,
    TransitionSample, WindowedNumericDeltaModel,
};

const ACTIONS: [&str; 3] = ["evade", "guard", "counter"];
const HEALTH_FIELD: &str = "health";
const EPISODE_STEPS: usize = 240;
const SEEDS: u64 = 64;
const WINDOW_SIZES: [usize; 6] = [4, 8, 12, 20, 32, 48];
const EXPLORATION_RATES: [f64; 3] = [0.05, 0.12, 0.20];
const INITIAL_HEALTH: f64 = 4.0;

#[derive(Clone, Copy)]
struct OpponentPhase {
    duration: usize,
    preferred_action: usize,
}

const OPPONENT: [OpponentPhase; 6] = [
    OpponentPhase {
        duration: 23,
        preferred_action: 0,
    },
    OpponentPhase {
        duration: 41,
        preferred_action: 1,
    },
    OpponentPhase {
        duration: 29,
        preferred_action: 2,
    },
    OpponentPhase {
        duration: 37,
        preferred_action: 0,
    },
    OpponentPhase {
        duration: 31,
        preferred_action: 2,
    },
    OpponentPhase {
        duration: 43,
        preferred_action: 1,
    },
];

#[derive(Clone, Copy, Default)]
struct Metrics {
    survival_steps: usize,
    reward: f64,
    prediction_absolute_error: f64,
    predictions: usize,
    phase_recovery_steps: usize,
    phase_recoveries: usize,
    phase_changes: usize,
    change_detections: usize,
    window_size_sum: usize,
    window_observations: usize,
}

#[derive(Clone, Copy)]
enum LearnerKind {
    Cumulative,
    FixedWindow,
    Adaptive,
}

struct Agent {
    model: NumericDeltaModel,
    recent_model: Option<WindowedNumericDeltaModel>,
    adaptive_model: Option<AdaptiveNumericDeltaModel>,
    exploration_rate: f64,
    rng: u64,
}

impl Agent {
    fn new(
        seed: u64,
        learner: LearnerKind,
        recent_window: usize,
        adaptive_policy: AdaptiveLearningPolicy,
        exploration_rate: f64,
    ) -> Self {
        let recent_model = match learner {
            LearnerKind::FixedWindow => Some(
                WindowedNumericDeltaModel::new(recent_window)
                    .expect("positive recent window configured by benchmark"),
            ),
            LearnerKind::Cumulative | LearnerKind::Adaptive => None,
        };
        let adaptive_model = match learner {
            LearnerKind::Adaptive => Some(
                AdaptiveNumericDeltaModel::new(adaptive_policy)
                    .expect("valid adaptive policy configured by benchmark"),
            ),
            LearnerKind::Cumulative | LearnerKind::FixedWindow => None,
        };
        Self {
            model: NumericDeltaModel::default(),
            recent_model,
            adaptive_model,
            exploration_rate,
            rng: seed.max(1),
        }
    }

    fn random(&mut self) -> u64 {
        // Deterministic xorshift generator: this is an experiment fixture, not
        // a source of security-sensitive randomness.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn predict_delta(&self, action: &str, state: &DataValue) -> Option<f64> {
        let predictions = if let Some(model) = &self.adaptive_model {
            model.predict_next(action, state).ok()?
        } else if let Some(model) = &self.recent_model {
            model.predict_next(action, state).ok()?
        } else {
            self.model.predict_next(action, state).ok()?
        };
        predictions
            .get(HEALTH_FIELD)
            .map(|estimate| estimate.mean_delta)
    }

    fn choose_action(&mut self, health: f64) -> usize {
        if self.random_unit() < self.exploration_rate {
            return (self.random() as usize) % ACTIONS.len();
        }

        let state = DataValue::object([(HEALTH_FIELD.to_owned(), DataValue::Number(health))]);
        let mut best: Option<(usize, f64)> = None;
        for (index, action) in ACTIONS.iter().enumerate() {
            let Some(estimate) = self.predict_delta(action, &state) else {
                // Optimism under ignorance: sample an unmeasured action before
                // exploiting a measured one.
                return index;
            };
            if best.map_or(true, |(_, score)| estimate > score) {
                best = Some((index, estimate));
            }
        }
        best.map(|(index, _)| index).unwrap_or(0)
    }

    fn random_unit(&mut self) -> f64 {
        (self.random() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn observe(&mut self, sample: TransitionSample) -> bool {
        if let Some(model) = &mut self.adaptive_model {
            model
                .observe_transition(&sample)
                .expect("finite survival transition")
                .change_detected
        } else if let Some(model) = &mut self.recent_model {
            model
                .observe_transition(&sample)
                .expect("finite survival transition");
            false
        } else {
            self.model
                .observe_transition(&sample)
                .expect("finite survival transition");
            false
        }
    }

    fn window_size(&self) -> Option<usize> {
        self.adaptive_model
            .as_ref()
            .map(AdaptiveNumericDeltaModel::window_size)
            .or_else(|| self.recent_model.as_ref().map(WindowedNumericDeltaModel::window_size))
    }
}

fn opponent_phase(step: usize) -> (usize, usize) {
    let cycle: usize = OPPONENT.iter().map(|phase| phase.duration).sum();
    let mut offset = step % cycle;
    for (index, phase) in OPPONENT.iter().enumerate() {
        if offset < phase.duration {
            return (index, phase.preferred_action);
        }
        offset -= phase.duration;
    }
    unreachable!("phase schedule is nonempty")
}

fn survival_delta(action: usize, preferred: usize) -> f64 {
    let distance = (action + ACTIONS.len() - preferred) % ACTIONS.len();
    match distance {
        0 => 0.75,
        1 => -0.20,
        _ => -1.15,
    }
}

fn run_episode(
    seed: u64,
    learner: LearnerKind,
    recent_window: usize,
    adaptive_policy: AdaptiveLearningPolicy,
    exploration_rate: f64,
) -> Metrics {
    let mut agent = Agent::new(
        seed,
        learner,
        recent_window,
        adaptive_policy,
        exploration_rate,
    );
    let mut metrics = Metrics::default();
    let mut health = INITIAL_HEALTH;
    let mut previous_phase = 0;
    let mut recovering = false;
    let mut recovery_steps = 0;
    let schedule_cycle: usize = OPPONENT.iter().map(|phase| phase.duration).sum();
    let schedule_offset = (seed % schedule_cycle as u64) as usize;

    for step in 0..EPISODE_STEPS {
        let (phase, preferred_action) = opponent_phase(step + schedule_offset);
        if step > 0 && phase != previous_phase {
            metrics.phase_changes += 1;
            recovering = true;
            recovery_steps = 0;
        }
        previous_phase = phase;

        let action = agent.choose_action(health);
        let delta = survival_delta(action, preferred_action);
        let next_health = health + delta;
        let state = DataValue::object([(HEALTH_FIELD.to_owned(), DataValue::Number(health))]);
        let next_state =
            DataValue::object([(HEALTH_FIELD.to_owned(), DataValue::Number(next_health))]);

        if let Some(predicted_delta) = agent.predict_delta(ACTIONS[action], &state) {
            metrics.prediction_absolute_error += (predicted_delta - delta).abs();
            metrics.predictions += 1;
        }

        if agent.observe(TransitionSample {
            action_id: ACTIONS[action].to_owned(),
            before: state,
            after: next_state,
        }) {
            metrics.change_detections += 1;
        }
        if let Some(window_size) = agent.window_size() {
            metrics.window_size_sum += window_size;
            metrics.window_observations += 1;
        }

        health = next_health;
        metrics.reward += delta;
        metrics.survival_steps = step + 1;

        if recovering {
            recovery_steps += 1;
            if action == preferred_action {
                metrics.phase_recovery_steps += recovery_steps;
                metrics.phase_recoveries += 1;
                recovering = false;
            }
        }
        if health <= 0.0 {
            break;
        }
    }
    metrics
}

fn summarize(
    learner: LearnerKind,
    recent_window: usize,
    adaptive_policy: AdaptiveLearningPolicy,
    exploration_rate: f64,
) -> Metrics {
    let mut total = Metrics::default();
    for seed in 1..=SEEDS {
        let metrics = run_episode(
            seed.wrapping_mul(0x9E37_79B9),
            learner,
            recent_window,
            adaptive_policy,
            exploration_rate,
        );
        total.survival_steps += metrics.survival_steps;
        total.reward += metrics.reward;
        total.prediction_absolute_error += metrics.prediction_absolute_error;
        total.predictions += metrics.predictions;
        total.phase_recovery_steps += metrics.phase_recovery_steps;
        total.phase_recoveries += metrics.phase_recoveries;
        total.phase_changes += metrics.phase_changes;
        total.change_detections += metrics.change_detections;
        total.window_size_sum += metrics.window_size_sum;
        total.window_observations += metrics.window_observations;
    }
    total
}

fn main() {
    let adaptive_policy = AdaptiveLearningPolicy {
        min_window_size: 4,
        max_window_size: 32,
        stable_observations_to_expand: 12,
        minimum_model_samples: 2,
        minimum_residual_samples: 12,
        consecutive_outliers: 3,
        change_threshold_sigma: 4.0,
    };
    println!("Adversarial survival experiment");
    println!("seeds={SEEDS}, horizon={EPISODE_STEPS}, hidden irregular opponent schedule");
    println!("no action, energy, or compute budget; survival ends only at zero health");
    println!("adaptive policy: minimum 4, maximum 32, expansion after 12 stable predictions");
    println!();
    println!("explore learner      survival  full-horizon  prediction  recovery  recovery  changes  avg-window");
    println!("rate    strategy     steps      survivors     MAE         success   lag");

    for exploration_rate in EXPLORATION_RATES {
        let baseline = summarize(
            LearnerKind::Cumulative,
            usize::MAX,
            adaptive_policy,
            exploration_rate,
        );
        let fixed_four = summarize(
            LearnerKind::FixedWindow,
            WINDOW_SIZES[0],
            adaptive_policy,
            exploration_rate,
        );
        let adaptive = summarize(
            LearnerKind::Adaptive,
            WINDOW_SIZES[0],
            adaptive_policy,
            exploration_rate,
        );
        assert!(
            fixed_four.survival_steps > baseline.survival_steps,
            "the four-transition baseline must out-survive the cumulative learner"
        );
        assert!(
            count_survivors(
                LearnerKind::FixedWindow,
                WINDOW_SIZES[0],
                adaptive_policy,
                exploration_rate,
            ) * 10
                >= SEEDS as usize * 9,
            "the fixed recent-window learner must survive at least 90% of benchmark episodes"
        );
        print_row(
            exploration_rate,
            "cumulative",
            baseline,
            count_survivors(
                LearnerKind::Cumulative,
                usize::MAX,
                adaptive_policy,
                exploration_rate,
            ),
        );
        print_row(
            exploration_rate,
            "fixed-4",
            fixed_four,
            count_survivors(
                LearnerKind::FixedWindow,
                WINDOW_SIZES[0],
                adaptive_policy,
                exploration_rate,
            ),
        );
        print_row(
            exploration_rate,
            "adaptive",
            adaptive,
            count_survivors(
                LearnerKind::Adaptive,
                WINDOW_SIZES[0],
                adaptive_policy,
                exploration_rate,
            ),
        );
        for window in WINDOW_SIZES.iter().copied().skip(1) {
            let fixed = summarize(
                LearnerKind::FixedWindow,
                window,
                adaptive_policy,
                exploration_rate,
            );
            print_row(
                exploration_rate,
                &format!("fixed-{window}"),
                fixed,
                count_survivors(
                    LearnerKind::FixedWindow,
                    window,
                    adaptive_policy,
                    exploration_rate,
                ),
            );
        }
        println!();
    }
}

fn print_row(
    exploration_rate: f64,
    learner: &str,
    metrics: Metrics,
    full_horizon_survivors: usize,
) {
    let prediction_mae = metrics.prediction_absolute_error / metrics.predictions.max(1) as f64;
    let recovery_rate = metrics.phase_recoveries as f64 / metrics.phase_changes.max(1) as f64;
    let recovery_lag = metrics.phase_recovery_steps as f64 / metrics.phase_recoveries.max(1) as f64;
    let average_window = metrics.window_size_sum as f64 / metrics.window_observations.max(1) as f64;
    println!(
        "{:<8.2} {:<12} {:<10.2} {:<13} {:<11.4} {:<9.2} {:<9.2} {:<8.2} {:.2}",
        exploration_rate,
        learner,
        metrics.survival_steps as f64 / SEEDS as f64,
        full_horizon_survivors,
        prediction_mae,
        recovery_rate,
        recovery_lag,
        metrics.change_detections as f64 / SEEDS as f64,
        average_window
    );
}

fn count_survivors(
    learner: LearnerKind,
    recent_window: usize,
    adaptive_policy: AdaptiveLearningPolicy,
    exploration_rate: f64,
) -> usize {
    (1..=SEEDS)
        .filter(|seed| {
            run_episode(
                seed.wrapping_mul(0x9E37_79B9),
                learner,
                recent_window,
                adaptive_policy,
                exploration_rate,
            )
            .survival_steps
                == EPISODE_STEPS
        })
        .count()
}
