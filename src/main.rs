use std::path::Path;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    optim::AdamConfig,
};
use model::ModelConfig;
use training::{train, TrainingConfig};

pub mod data;
pub mod model;
pub mod training;
pub mod module;

fn main() {
    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = WgpuDevice::default();
    let artifact_dir = Path::new("artifact");

    train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 128), AdamConfig::new()).with_epoch_count(20),
        device,
    );
}
