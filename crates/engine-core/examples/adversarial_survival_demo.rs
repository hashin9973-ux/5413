//! Reproducible adversarial survival experiment for online model adaptation.
//!
//! The opponent changes its counter-strategy on a hidden schedule. The agent
//! observes only its own survival delta, not the opponent's phase. There is no
//! action, energy, or compute budget; survival ends only when health reaches zero.
//! The experiment compares a cumulative NumericDeltaModel with a recent-window
//! variant that can discard stale evidence after opponent strategy changes.

use engine_core::{DataValue, NumericDeltaModel, TransitionSample};
use std::collections::VecDeque;

const ACTIONS: [&str; 3] = ["evade", "guard", "counter"];
const HEALTH_FIELD: &str = "health";
const EPISODE_STEPS: usize = 240;
const SEEDS: u64 = 64;
const RECENT_WINDOW: usize = 12;
const EXPLORATION_RATE: f64 = 0.12;
const INITIAL_HEALTH: f64 = 4.0;

#[derive(Clone, Copy)]
struct OpponentPhase {
    duration: usize,
    preferred_action: usize,
}

const OPPONENT: [OpponentPhase; 6] = [
    OpponentPhase { duration: 23, preferred_action: 0 },
    OpponentPhase { duration: 41, preferred_action: 1 },
    OpponentPhase { duration: 29, preferred_action: 2 },
    OpponentPhase { duration: 37, preferred_action: 0 },
    OpponentPhase { duration: 31, preferred_action: 2 },
    OpponentPhase { duration: 43, preferred_action: 1 },
];

#[derive(Clone, Copy, Default)]
struct Metrics {
    survival_steps: usize,
    reward: f64,
    prediction_absolute_error: f64,
    predictions: usize,
    phase_recovery_steps: usize,
    phase_changes: usize,
}

struct Agent {
    model: NumericDeltaModel,
    history: VecDeque<TransitionSample>,
    adaptive: bool,
    rng: u64,
}

impl Agent {
    fn new(seed: u64, adaptive: bool) -> Self {
        Self {
            model: NumericDeltaModel::default(),
            history: VecDeque::new(),
            adaptive,
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

    fn choose_action(&mut self, health: f64) -> usize {
        if self.random_unit() < EXPLORATION_RATE {
            return (self.random() as usize) % ACTIONS.len();
        }

        let state = DataValue::object([(HEALTH_FIELD.to_owned(), DataValue::Number(health))]);
        let mut best: Option<(usize, f64)> = None;
        for (index, action) in ACTIONS.iter().enumerate() {
            let estimate = self
                .model
                .predict_next(action, &state)
                .ok()
                .and_then(|values| values.get(HEALTH_FIELD).copied());
            let Some(estimate) = estimate else {
                // Optimism under ignorance: sample an unmeasured action before
                // exploiting a measured one.
                return index;
            };
            if best.is_none_or(|(_, score)| estimate.mean_delta > score) {
                best = Some((index, estimate.mean_delta));
            }
        }
        best.map(|(index, _)| index).unwrap_or(0)
    }

    fn random_unit(&mut self) -> f64 {
        (self.random() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn observe(&mut self, sample: TransitionSample) {
        self.history.push_back(sample.clone());
        while self.history.len() > RECENT_WINDOW {
            self.history.pop_front();
        }
        if self.adaptive {
            self.model = NumericDeltaModel::default();
            for recent in &self.history {
                self.model
                    .observe_transition(recent)
                    .expect("finite survival transition");
            }
        } else {
            self.model
                .observe_transition(&sample)
                .expect("finite survival transition");
        }
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

fn run_episode(seed: u64, adaptive: bool) -> Metrics {
    let mut agent = Agent::new(seed, adaptive);
    let mut metrics = Metrics::default();
    let mut health = INITIAL_HEALTH;
    let mut previous_phase = 0;
    let mut recovering = false;
    let mut recovery_steps = 0;

    for step in 0..EPISODE_STEPS {
        let (phase, preferred_action) = opponent_phase(step);
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

        if let Ok(predictions) = agent.model.predict_next(ACTIONS[action], &state) {
            if let Some(prediction) = predictions.get(HEALTH_FIELD) {
                metrics.prediction_absolute_error += (prediction.mean_delta - delta).abs();
                metrics.predictions += 1;
            }
        }

        agent.observe(TransitionSample {
            action_id: ACTIONS[action].to_owned(),
            before: state,
            after: next_state,
        });

        health = next_health;
        metrics.reward += delta;
        metrics.survival_steps = step + 1;

        if recovering {
            recovery_steps += 1;
            if action == preferred_action {
                metrics.phase_recovery_steps += recovery_steps;
                recovering = false;
            }
        }
        if health <= 0.0 {
            break;
        }
    }
    metrics
}

fn summarize(adaptive: bool) -> Metrics {
    let mut total = Metrics::default();
    for seed in 1..=SEEDS {
        let metrics = run_episode(seed.wrapping_mul(0x9E37_79B9), adaptive);
        total.survival_steps += metrics.survival_steps;
        total.reward += metrics.reward;
        total.prediction_absolute_error += metrics.prediction_absolute_error;
        total.predictions += metrics.predictions;
        total.phase_recovery_steps += metrics.phase_recovery_steps;
        total.phase_changes += metrics.phase_changes;
    }
    total
}

fn main() {
    let baseline = summarize(false);
    let adaptive = summarize(true);
    let average = |value: usize| value as f64 / SEEDS as f64;
    let prediction_mae = |metrics: Metrics| {
        metrics.prediction_absolute_error / metrics.predictions.max(1) as f64
    };
    let recovery = |metrics: Metrics| {
        metrics.phase_recovery_steps as f64 / metrics.phase_changes.max(1) as f64
    };

    println!("Adversarial survival experiment");
    println!("seeds={SEEDS}, horizon={EPISODE_STEPS}, recent_window={RECENT_WINDOW}");
    println!("opponent=hidden rotating counter-strategy; health is the only survival condition");
    println!();
    println!("metric                         cumulative      recent-window");
    println!(
        "mean survival steps            {:>10.2}      {:>10.2}",
        average(baseline.survival_steps),
        average(adaptive.survival_steps)
    );
    println!(
        "mean cumulative survival delta {:>10.2}      {:>10.2}",
        baseline.reward / SEEDS as f64,
        adaptive.reward / SEEDS as f64
    );
    println!(
        "prediction MAE                 {:>10.4}      {:>10.4}",
        prediction_mae(baseline),
        prediction_mae(adaptive)
    );
    println!(
        "mean phase recovery steps      {:>10.2}      {:>10.2}",
        recovery(baseline),
        recovery(adaptive)
    );
    println!(
        "episodes survived to horizon   {:>10}      {:>10}",
        count_survivors(false),
        count_survivors(true)
    );
}

fn count_survivors(adaptive: bool) -> usize {
    (1..=SEEDS)
        .filter(|seed| {
            run_episode(seed.wrapping_mul(0x9E37_79B9), adaptive).survival_steps == EPISODE_STEPS
        })
        .count()
}
