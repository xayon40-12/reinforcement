use std::ops::RangeInclusive;

use super::{
    BoundedNetwork, Float, ForwardNetwork, JoinNetwork, Network, activation::Activation,
    layer::Layer,
};

#[derive(Default)]
pub struct Layers<const NI: usize, const NH: usize, A: Activation, O: JoinNetwork<NH>> {
    layer_in: Layer<NI, NH, A>,
    layer_out: O,
}
pub type LS<const NI: usize, const NH: usize, A, O> = Layers<NI, NH, A, O>;

impl<const NI: usize, const NH: usize, A: Activation, O: JoinNetwork<NH>> JoinNetwork<NI>
    for Layers<NI, NH, A, O>
{
    type OutA = O::OutA;
}
impl<const NI: usize, const NH: usize, const NO: usize, A: Activation, O: Network<NH, NO>>
    ForwardNetwork<NI, NO> for Layers<NI, NH, A, O>
{
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO] {
        self.layer_out.forward(self.layer_in.forward(input))
    }
}
impl<const NI: usize, const NH: usize, const NO: usize, A: Activation, O: Network<NH, NO>>
    Network<NI, NO> for Layers<NI, NH, A, O>
{
    fn randomize(&mut self) {
        self.layer_in.randomize();
        self.layer_out.randomize();
    }
    fn update_gradient(&mut self, relaxation: Float, delta: [Float; NO]) -> [Float; NI] {
        self.layer_in.update_gradient(
            relaxation,
            self.layer_out.update_gradient(relaxation, delta),
        )
    }
    fn reset_gradient(&mut self) {
        self.layer_in.reset_gradient();
        self.layer_out.reset_gradient();
    }
    fn apply_gradient(&mut self, alpha: Float) {
        self.layer_in.apply_gradient(alpha);
        self.layer_out.apply_gradient(alpha);
    }
    fn norm2_gradient(&self) -> Float {
        self.layer_in.norm2_gradient() + self.layer_out.norm2_gradient()
    }
    fn rescale_gradient(&mut self, a: Float) {
        self.layer_in.rescale_gradient(a);
        self.layer_out.rescale_gradient(a);
    }
}
impl<const NI: usize, const NH: usize, const NO: usize, A: Activation, O: BoundedNetwork<NH, NO>>
    BoundedNetwork<NI, NO> for Layers<NI, NH, A, O>
{
    fn output_ranges(&self) -> [RangeInclusive<Float>; NO] {
        self.layer_out.output_ranges()
    }
}
