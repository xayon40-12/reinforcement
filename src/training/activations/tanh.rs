use num::Float;

use crate::training::{Activation, BackProp, Eval, Weights};

pub struct Tanh {
    inputs: usize,
}
impl Tanh {
    pub fn new(inputs: usize) -> Self {
        Tanh { inputs }
    }
    pub fn layer<T: Float>(inputs: usize) -> (usize, Box<dyn Activation<T>>) {
        (inputs, Box::new(Tanh { inputs }))
    }
}
impl<T: Float> Weights<T> for Tanh {
    fn weights_len(&self) -> usize {
        0
    }
}
impl<T: Float> Eval<T> for Tanh {
    fn state_len(&self) -> usize {
        self.inputs
    }
    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        debug_assert!(
            input.len() == self.inputs && weights.len() == 0 && state.len() == self.inputs
        );
        state
            .iter_mut()
            .zip(input.iter())
            .for_each(|(s, i)| *s = i.tanh());
    }
    fn input_len(&self) -> usize {
        self.inputs
    }
    fn output_len(&self) -> usize {
        self.inputs
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        debug_assert!(state.len() == self.inputs);
        state
    }
    fn output_mut<'a>(&self, state: &'a mut [T]) -> &'a mut [T] {
        debug_assert!(state.len() == self.inputs);
        state
    }
}
impl<T: Float> BackProp<T> for Tanh {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &mut [T],
        gradient: &mut [T],
    ) {
        debug_assert!(input.len() == self.inputs, "Tanh input");
        debug_assert!(weights.len() == 0, "Tanh, weights");
        debug_assert!(state.len() == self.inputs, "Tanh state");
        debug_assert!(gradient.len() == 0, "Tanh gradient");
        debug_assert!(front.len() == self.inputs, "Tanh front");
        debug_assert!(back.len() == self.inputs, "Tanh back");
        front
            .iter_mut()
            .zip(input.iter())
            .zip(back.iter())
            .for_each(|((f, i), b)| *f = (T::one() - i.tanh().powi(2)) * *b);
    }
}
impl<T: Float> Activation<T> for Tanh {
    fn range(&self) -> (Option<T>, Option<T>) {
        (None, None)
    }
}
