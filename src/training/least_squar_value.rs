use num::Float;

use super::{BackProp, Eval, Gradient, Value, Weights, mlp::MLP};

pub struct LeastSquareValue<T: Float> {
    mlp: MLP<T>,
}
impl<T: Float> LeastSquareValue<T> {
    pub fn new(mlp: MLP<T>) -> Self {
        debug_assert!(
            mlp.output_len() == 1,
            "The LeastSquareValue MLP must output a single value"
        );
        LeastSquareValue { mlp }
    }
}

impl<T: Float> Weights<T> for LeastSquareValue<T> {
    fn weights_len(&self) -> usize {
        self.mlp.weights_len()
    }
}
impl<T: Float> Eval<T> for LeastSquareValue<T> {
    fn state_len(&self) -> usize {
        self.mlp.state_len() + 2 * self.mlp.min_back_front_len() + 1
    }
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        self.mlp
            .eval(input, weights, &mut state[..self.mlp.state_len()]);
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        self.mlp.output(&state[..self.mlp.state_len()])
    }
}
impl<T: Float> Gradient<T> for LeastSquareValue<T> {
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &mut [T], gradient: &mut [T]) {
        let value = self.value(state);
        let (state, tmp) = state.split_at_mut(self.mlp.state_len());
        let (back, tmp) = tmp.split_at_mut(self.mlp.min_back_front_len());
        let (front, target) = tmp.split_at_mut(self.mlp.min_back_front_len());
        back[0] = value - target[0];
        self.mlp
            .back_prop(input, weights, state, front, back, gradient);
    }
}

impl<T: Float> Value<T> for LeastSquareValue<T> {
    fn set_target(&mut self, target: T, state: &mut [T]) {
        state[self.state_len() - 1] = target;
    }
    fn target(&self, state: &[T]) -> T {
        state[self.state_len() - 1]
    }
    fn value(&self, state: &[T]) -> T {
        self.output(state)[0]
    }
}
