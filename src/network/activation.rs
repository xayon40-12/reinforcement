use std::ops::RangeInclusive;

use super::Float;

pub trait Activation: Default + Copy {
    fn apply(&self, x: Float) -> Float;
    fn derivative(&self, x: Float) -> Float;
}

pub trait BoundedActivation: Activation {
    fn range(&self) -> RangeInclusive<Float>;
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
pub struct SigmoidSim;
impl Activation for SigmoidSim {
    fn apply(&self, input: Float) -> Float {
        2.0 * (1.0 + (-input).exp()).recip() - 1.0
    }
    fn derivative(&self, input: Float) -> Float {
        let e = (-input).exp();
        2.0 * e / (1.0 + e).powi(2)
    }
}
impl BoundedActivation for SigmoidSim {
    fn range(&self) -> RangeInclusive<Float> {
        -1.0..=1.0
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn apply(&self, input: Float) -> Float {
        (1.0 + (-input).exp()).recip()
    }
    fn derivative(&self, input: Float) -> Float {
        let e = (-input).exp();
        e / (1.0 + e).powi(2)
    }
}
impl BoundedActivation for Sigmoid {
    fn range(&self) -> RangeInclusive<Float> {
        0.0..=1.0
    }
}
