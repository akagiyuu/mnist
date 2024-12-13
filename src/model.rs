use burn::{
    nn::{Dropout, Linear, Relu},
    prelude::*,
    tensor::{activation::log_softmax, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use nn::{
    loss::CrossEntropyLossConfig,
    pool::{MaxPool2d, MaxPool2dConfig},
    DropoutConfig, LinearConfig,
};

use crate::{
    data::MnistBatch,
    module::{SeparableConv2D, SeparableConv2DConfig},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv: [SeparableConv2D<B>; 2],
    pool: MaxPool2d,
    linear: [Linear<B>; 2],

    dropout: [Dropout; 2],
    relu: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let mut x = images.reshape([batch_size, 1, height, width]);

        for conv in self.conv.iter() {
            x = conv.forward(x);
            x = self.relu.forward(x);
        }
        let x = self.pool.forward(x);
        let x = self.dropout[0].forward(x);

        let x = x.reshape([batch_size, 16 * 22 * 22]);
        let x = self.linear[0].forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout[1].forward(x);
        let x = self.linear[1].forward(x);
        log_softmax(x, 1)
    }

    pub fn foward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.foward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.foward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    class_count: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv: [
                SeparableConv2DConfig::new([1, 8], [3, 3]).init(device),
                SeparableConv2DConfig::new([8, 16], [3, 3]).init(device),
            ],
            pool: MaxPool2dConfig::new([3, 3]).init(),
            linear: [
                LinearConfig::new(16 * 22 * 22, self.hidden_size).init(device),
                LinearConfig::new(self.hidden_size, self.class_count).init(device),
            ],
            relu: Relu::new(),
            dropout: [
                DropoutConfig::new(0.25).init(),
                DropoutConfig::new(0.5).init(),
            ],
        }
    }
}
