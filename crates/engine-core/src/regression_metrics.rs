//! Numerically stable accumulators for prediction and regression metrics.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RegressionErrorSummary {
    pub mean_absolute_error: f64,
    pub root_mean_squared_error: f64,
}

#[derive(Debug)]
pub(crate) struct RegressionErrorAccumulator {
    numeric_fields: usize,
    metrics_representable: bool,
    absolute_error_mean: OnlineMean,
    squared_error_sum: ScaledSumSquares,
}

impl Default for RegressionErrorAccumulator {
    fn default() -> Self {
        Self {
            numeric_fields: 0,
            metrics_representable: true,
            absolute_error_mean: OnlineMean::default(),
            squared_error_sum: ScaledSumSquares::default(),
        }
    }
}

impl RegressionErrorAccumulator {
    /// Records a matched prediction/observation pair and returns its absolute error.
    /// Returns `None` when the difference itself is outside the finite `f64` range.
    pub(crate) fn record_difference(&mut self, predicted: f64, observed: f64) -> Option<f64> {
        self.numeric_fields += 1;
        let error = predicted - observed;
        if !error.is_finite() {
            self.metrics_representable = false;
            return None;
        }

        let absolute_error = error.abs();
        if !self.absolute_error_mean.push_nonnegative(absolute_error)
            || !self.squared_error_sum.push_nonnegative(absolute_error)
        {
            self.metrics_representable = false;
            return None;
        }

        Some(absolute_error)
    }

    pub(crate) fn numeric_fields(&self) -> usize {
        self.numeric_fields
    }

    pub(crate) fn finish(&self) -> Option<RegressionErrorSummary> {
        if self.numeric_fields == 0 || !self.metrics_representable {
            return None;
        }

        Some(RegressionErrorSummary {
            mean_absolute_error: self.absolute_error_mean.mean()?,
            root_mean_squared_error: self.squared_error_sum.root_mean_square()?,
        })
    }
}

/// Online mean for non-negative finite measurements such as absolute errors
/// and predicted standard deviations.
#[derive(Debug)]
pub(crate) struct OnlineMean {
    count: usize,
    mean: f64,
    valid: bool,
}

impl Default for OnlineMean {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            valid: true,
        }
    }
}

impl OnlineMean {
    pub(crate) fn push_nonnegative(&mut self, value: f64) -> bool {
        if !value.is_finite() || value < 0.0 {
            self.valid = false;
            return false;
        }

        self.count += 1;
        if self.count == 1 {
            self.mean = value;
        } else {
            // Both values are non-negative, so their difference is representable.
            self.mean += (value - self.mean) / self.count as f64;
        }

        if !self.mean.is_finite() {
            self.valid = false;
            return false;
        }
        true
    }

    pub(crate) fn mean(&self) -> Option<f64> {
        (self.valid && self.count > 0).then_some(self.mean)
    }
}

/// Scaled sum-of-squares accumulator, following the stable LASSQ pattern.
#[derive(Debug)]
struct ScaledSumSquares {
    count: usize,
    scale: f64,
    scaled_sum: f64,
    valid: bool,
}

impl Default for ScaledSumSquares {
    fn default() -> Self {
        Self {
            count: 0,
            scale: 0.0,
            scaled_sum: 0.0,
            valid: true,
        }
    }
}

impl ScaledSumSquares {
    fn push_nonnegative(&mut self, value: f64) -> bool {
        if !value.is_finite() || value < 0.0 {
            self.valid = false;
            return false;
        }

        self.count += 1;
        if value == 0.0 {
            return true;
        }

        if value > self.scale {
            let ratio = if self.scale == 0.0 {
                0.0
            } else {
                self.scale / value
            };
            self.scaled_sum = 1.0 + self.scaled_sum * ratio * ratio;
            self.scale = value;
        } else {
            let ratio = value / self.scale;
            self.scaled_sum += ratio * ratio;
        }

        if !self.scaled_sum.is_finite() {
            self.valid = false;
            return false;
        }
        true
    }

    fn root_mean_square(&self) -> Option<f64> {
        if !self.valid || self.count == 0 {
            return None;
        }
        if self.scale == 0.0 {
            return Some(0.0);
        }

        let result = self.scale * (self.scaled_sum / self.count as f64).sqrt();
        result.is_finite().then_some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics_avoid_overflow_in_mean_and_squared_sum() {
        let mut metrics = RegressionErrorAccumulator::default();
        assert_eq!(metrics.record_difference(1.0e308, 0.0), Some(1.0e308));
        assert_eq!(metrics.record_difference(1.0e308, 0.0), Some(1.0e308));

        let summary = metrics.finish().expect("finite summary");
        assert_eq!(metrics.numeric_fields(), 2);
        assert!((summary.mean_absolute_error / 1.0e308 - 1.0).abs() < 1.0e-12);
        assert!((summary.root_mean_squared_error / 1.0e308 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn regression_metrics_reject_unrepresentable_differences() {
        let mut metrics = RegressionErrorAccumulator::default();
        assert_eq!(metrics.record_difference(f64::MAX, -f64::MAX), None);
        assert_eq!(metrics.numeric_fields(), 1);
        assert_eq!(metrics.finish(), None);
    }
}
