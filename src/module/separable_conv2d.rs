use burn::{nn::conv::Conv2d, prelude::*};
use nn::conv::Conv2dConfig;

#[derive(Module, Debug)]
pub struct SeparableConv2D<B: Backend> {
    depthwise: Conv2d<B>,
    pointwise: Conv2d<B>,
}

impl<B: Backend> SeparableConv2D<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(x);
        self.pointwise.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct SeparableConv2DConfig {
    channels: [usize; 2],
    kernel_size: [usize; 2],
}

impl SeparableConv2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SeparableConv2D<B> {
        SeparableConv2D {
            depthwise: Conv2dConfig::new(
                [self.channels[0], self.channels[0]],
                self.kernel_size,
            )
            .with_groups(self.channels[0])
            .with_padding(nn::PaddingConfig2d::Same)
            .init(device),
            pointwise: Conv2dConfig::new([self.channels[0], self.channels[1]], [1, 1])
                .init(device),
        }
    }
}
