use num::Float;

pub mod activations;
pub mod layer_matrix;
pub mod mlp;
pub mod optimizers;
pub mod policies;
pub mod trainer;

pub trait Weights<T: Float> {
    fn weights_len(&self) -> usize;
    fn empty_weights(&self) -> Box<[T]> {
        vec![T::zero(); self.weights_len()].into_boxed_slice()
    }
}
pub trait Eval<T: Float>: Weights<T> {
    fn state_len(&self) -> usize;
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]);
    fn output<'a>(&self, state: &'a [T]) -> &'a [T];
    fn empty_state(&self) -> Box<[T]> {
        vec![T::zero(); self.state_len()].into_boxed_slice()
    }
}
pub trait Gradient<T: Float>: Eval<T> {
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &mut [T], gradient: &mut [T]);
}
pub trait BackProp<T: Float>: Eval<T> {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &mut [T],
        gradient: &mut [T],
    );
}
/// It is `Gradient` with respect to the scalar probability
pub trait StochasticPolicy<T: Float>: Gradient<T> {
    fn probability(&self, state: &[T]) -> T;
    fn stochastic_eval(&self, input: &[T], weights: &[T], state: &mut [T]);
    fn stochastic_output<'a>(&self, state: &'a [T]) -> &'a [T];
}

pub trait Activation<T: Float>: BackProp<T> {
    fn range(&self) -> (Option<T>, Option<T>);
}

pub trait Optimizer<T: Float> {
    fn step(&mut self, weights: &mut [T], gradient: &mut [T]);
}
