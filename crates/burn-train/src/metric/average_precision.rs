use core::f64;
use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Shape, Tensor};

/// [Average Precision](https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html) (also referred to as AP) for binary classification.
///
/// This metric calculates the average precision score for binary classification tasks.
/// It is particularly useful when dealing with imbalanced datasets.
#[derive(Default)]
pub struct AveragePrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

/// The [Average Precision metric](AveragePrecisionMetric) input type.
///
/// This struct holds the model outputs and the true target labels required to compute the average precision.
#[derive(new)]
pub struct AveragePrecisionInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AveragePrecisionMetric<B> {
    /// Creates a new instance of the `AveragePrecisionMetric`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the binary average precision score.
    ///
    /// # Arguments
    /// * `probabilities` - A tensor containing the predicted probabilities for the positive class.
    /// * `targets` - A tensor containing the true binary labels (0 or 1).
    ///
    /// # Returns
    /// The average precision score as an `f64`.
    fn binary_average_precision(
        &self,
        probabilities: &Tensor<B, 1>,
        targets: &Tensor<B, 1, Int>,
    ) -> f64 {
        let [n_samples] = probabilities.dims();
        let positives = targets.clone().sum().into_scalar().elem::<u64>() as usize;

        // Early return if we don't have both positive samples
        if positives == 0 {
            log::warn!("Metric cannot be computed because all target values are negative.");
            return 0.0;
        }

        let desc_score_indices = probabilities.clone().argsort_descending(0);

        let probabilities = probabilities.clone().select(0, desc_score_indices.clone());
        let targets = targets.clone().select(0, desc_score_indices.clone());

        let probabilities_cur = probabilities.clone().slice([0..n_samples - 1]);
        let probabilities_next = probabilities.clone().slice([1..n_samples]);
        let distinct_value_indices = (probabilities_next - probabilities_cur)
            .abs()
            .greater_elem(0)
            .nonzero()
            .into_iter()
            .next()
            .unwrap_or(Tensor::<B, 1, Int>::ones(
                Shape::new([0]),
                &probabilities.device(),
            ));
        let threshold_idxs = Tensor::cat(
            vec![
                distinct_value_indices.clone(),
                Tensor::full(
                    Shape::new([1]),
                    (n_samples - 1) as u64,
                    &distinct_value_indices.device(),
                ),
            ],
            0,
        );

        let tps = targets
            .clone()
            .cumsum(0)
            .select(0, threshold_idxs.clone())
            .float();
        let threshold_idxs = threshold_idxs.float();
        let fps = threshold_idxs - tps.clone() + 1;

        let precision = tps.clone().div(tps.clone() + fps);
        let positives = targets.clone().sum().float();
        let recall = tps.div(positives);
        let precision = precision.flip([0]);
        let recall = recall.flip([0]);

        let precision = Tensor::cat(
            vec![
                precision.clone(),
                Tensor::ones(Shape::new([1]), &precision.device()),
            ],
            0,
        );
        let recall = Tensor::cat(
            vec![
                recall.clone(),
                Tensor::zeros(Shape::new([1]), &recall.device()),
            ],
            0,
        );

        let n = precision.dims()[0];
        let recall_cur = recall.clone().slice([0..n - 1]);
        let recall_next = recall.clone().slice([1..n]);
        let precision_cur = precision.clone().slice([0..n - 1]);

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

    /// Updates the metric state with the given input and metadata.
    ///
    /// # Arguments
    /// * `input` - The input containing the model outputs and true labels.
    /// * `_metadata` - Additional metadata (not used in this implementation).
    ///
    /// # Returns
    /// A `MetricEntry` containing the updated metric value.
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
            100.0 * average_precision,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    /// Resets the metric state.
    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for AveragePrecisionMetric<B> {
    /// Returns the current value of the metric.
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
                    [0.46, 0.54],
                    [0.31, 0.69],
                    [0.32, 0.68],
                    [0.90, 0.10],
                    [0.84, 0.16],
                    [0.24, 0.76],
                    [0.75, 0.25],
                    [0.21, 0.79],
                    [0.43, 0.57],
                    [0.40, 0.60],
                ],
                &device,
            ),
            Tensor::from_data([0, 0, 0, 1, 1, 1, 0, 1, 1, 1], &device), // True labels
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value() - 73.7037).abs() < 1e-5);
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
        assert_eq!(metric.value(), 100.0); // Perfect AP
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
    fn test_average_precision_all_positives() {
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
        assert_eq!(metric.value(), 100.0);
    }

    #[test]
    fn test_average_precision_all_negatives() {
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
            Tensor::from_data([0, 0, 0, 0], &device), // All positive class
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
