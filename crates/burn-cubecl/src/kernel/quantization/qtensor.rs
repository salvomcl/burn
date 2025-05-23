#![allow(missing_docs)] // cube derive macros

use burn_tensor::quantization::{QuantizationMode, QuantizationScheme};
use cubecl::prelude::*;

/// Quantization parameters.
#[derive(CubeLaunch, CubeType)]
pub struct QParams {
    #[cube(comptime)]
    scheme: QuantizationScheme,
}

/// Quantized tensor representation.
pub type QTensor = Array<Line<u32>>;

#[cube]
impl QParams {
    /// Create a new quantization parameters instance.
    pub fn new(scheme: QuantizationScheme) -> Self {
        QParams { scheme }
    }

    /// Get the quantization parameters values.
    pub fn values(&self, tensor: &QTensor) -> (f32, i32) {
        let len = tensor.len();
        match comptime!(self.scheme) {
            // Symmetric quantization only contains the scaling factor as the last element
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, _) => {
                (f32::reinterpret(tensor[len - 1][tensor.line_size() - 1]), 0)
            }
        }
    }
}
