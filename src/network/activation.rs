use super::Float;

pub trait Activation: Default + Copy {
    fn apply(&self, x: Float) -> Float;
    fn derivative(&self, x: Float) -> Float;
    fn range(&self) -> (Option<Float>, Option<Float>) {
        (None, None)
    }
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
    fn range(&self) -> (Option<Float>, Option<Float>) {
        (Some(0.0), None)
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct SigmoidSim;
impl Activation for SigmoidSim {
    fn apply(&self, input: Float) -> Float {
        2.0 * (1.0 + (-0.5 * input).exp()).recip() - 1.0
    }
    fn derivative(&self, input: Float) -> Float {
        let e = (-0.5 * input).exp();
        if e.is_infinite() || e.is_nan() {
            0.0
        } else {
            e / (1.0 + e).powi(2)
        }
    }
    fn range(&self) -> (Option<Float>, Option<Float>) {
        (Some(-1.0), Some(1.0))
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
        if e.is_infinite() || e.is_nan() {
            0.0
        } else {
            e / (1.0 + e).powi(2)
        }
    }
    fn range(&self) -> (Option<Float>, Option<Float>) {
        (Some(0.0), Some(1.0))
    }
}
