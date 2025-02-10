use core::f64;
use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) (also referred to as AP) for binary classification.
#[derive(Default)]
pub struct AveragePrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

/// The [Average Precision metric](AveragePrecisionMetric) input type.
#[derive(new)]
pub struct AveragePrecisionInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AveragePrecisionMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    fn binary_average_precision(
        &self,
        probabilities: &Tensor<B, 1>,
        targets: &Tensor<B, 1, Int>,
    ) -> f64 {
        /*TODO do I need this?
        // Early return if we don't have both positive and negative samples
        if n_pos == 0 || n_pos == n {
            if n_pos == 0 {
                log::warn!("Metric cannot be computed because all target values are negative.")
            } else {
                log::warn!("Metric cannot be computed because all target values are positive.")
            }
            return 0.0;
        } */

        let [n_samples] = probabilities.dims();

        let desc_score_indices = probabilities.clone().argsort_descending(0);

        let targets = targets.clone().select(0, desc_score_indices.clone());

        let threshold_idxs = Tensor::arange(0..(n_samples as i64), &probabilities.device()).float();
        let tps = targets.clone().cumsum(0).float();
        let fps = threshold_idxs - tps.clone() + 1;

        let precision = tps.clone().div(tps.clone() + fps);
        let positives = targets.sum().float();
        let recall = tps.div(positives);
        let precision = precision.flip([0]);
        let recall = recall.flip([0]);

        let recall_cur = recall.clone().slice([0..n_samples - 1]);
        let recall_next = recall.slice([1..n_samples]);
        let precision_cur = precision.slice([0..n_samples - 1]);
        let average_precision = (-(recall_next - recall_cur) * precision_cur)
            .sum()
            .into_scalar()
            .elem::<f64>();
        average_precision
    }
}

impl<B: Backend> Metric for AveragePrecisionMetric<B> {
    const NAME: &'static str = "Average Precision";
    type Input = AveragePrecisionInput<B>;

    fn update(
        &mut self,
        input: &AveragePrecisionInput<B>,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let [batch_size, num_classes] = input.outputs.dims();

        assert_eq!(
            num_classes, 2,
            "Currently only binary classification is supported"
        );

        let probabilities = {
            let exponents = input.outputs.clone().exp();
            let sum = exponents.clone().sum_dim(1);
            (exponents / sum)
                .select(1, Tensor::arange(1..2, &input.outputs.device()))
                .squeeze(1)
        };

        let average_precision = self.binary_average_precision(&probabilities, &input.targets);

        self.state.update(
            100.0 * average_precision, // TODO, max 1?
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for AveragePrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_average_precision() {
        let device = Default::default();
        let mut metric = AveragePrecisionMetric::<TestBackend>::new();

        let input = AveragePrecisionInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.9], // High confidence positive
                    [0.7, 0.3], // Low confidence negative
                    [0.6, 0.4], // Low confidence negative
                    [0.2, 0.8], // High confidence positive
                ],
                &device,
            ),
            Tensor::from_data([1, 0, 0, 1], &device), // True labels
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 100.0);
    }

    #[test]
    fn test_average_precision_perfect_separation() {
        let device = Default::default();
        let mut metric = AveragePrecisionMetric::<TestBackend>::new();

        let input = AveragePrecisionInput::new(
            Tensor::from_data([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device),
            Tensor::from_data([1, 0, 0, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 100.0); // Perfect AUC
    }

    #[test]
    fn test_average_precision_random() {
        let device = Default::default();
        let mut metric = AveragePrecisionMetric::<TestBackend>::new();

        let input = AveragePrecisionInput::new(
            Tensor::from_data(
                [
                    [0.5, 0.5], // Random predictions
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
                &device,
            ),
            Tensor::from_data([1, 0, 0, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 50.0);
    }

    #[test]
    fn test_average_precision_all_one_class() {
        let device = Default::default();
        let mut metric = AveragePrecisionMetric::<TestBackend>::new();

        let input = AveragePrecisionInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.9], // All positives predictions
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6],
                ],
                &device,
            ),
            Tensor::from_data([1, 1, 1, 1], &device), // All positive class
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Currently only binary classification is supported")]
    fn test_average_precision_multiclass_error() {
        let device = Default::default();
        let mut metric = AveragePrecisionMetric::<TestBackend>::new();

        let input = AveragePrecisionInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.2, 0.7], // More than 2 classes not supported
                    [0.3, 0.5, 0.2],
                ],
                &device,
            ),
            Tensor::from_data([2, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
    }
}
