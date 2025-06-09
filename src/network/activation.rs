use super::Float;

pub trait Activation: Default + Copy {
    fn apply(&self, x: Float) -> Float;
    fn derivative(&self, x: Float) -> Float;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Id;
impl Activation for Id {
    fn apply(&self, input: Float) -> Float {
        input
    }
    fn derivative(&self, _input: Float) -> Float {
        1.0
    }
}
#[derive(Copy, Clone, Default, Debug)]
pub struct Relu;
impl Activation for Relu {
    fn apply(&self, input: Float) -> Float {
        input.max(0.0)
    }
    fn derivative(&self, input: Float) -> Float {
        input.signum().max(0.0)
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn apply(&self, input: Float) -> Float {
        2.0 * (1.0 + (-input).exp()).recip() - 1.0
    }
    fn derivative(&self, input: Float) -> Float {
        let e = (-input).exp();
        2.0 * e / (1.0 + e).powi(2)
    }
}
