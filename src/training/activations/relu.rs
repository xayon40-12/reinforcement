use num::Float;

use crate::training::{Activation, BackProp, Eval, Weights};

pub struct ReLu {
    inputs: usize,
}
impl ReLu {
    pub fn new(inputs: usize) -> Self {
        ReLu { inputs }
    }
    pub fn layer<T: Float>(inputs: usize) -> (usize, Box<dyn Activation<T>>) {
        (inputs, Box::new(ReLu { inputs }))
    }
}
impl<T: Float> Weights<T> for ReLu {
    fn weights_len(&self) -> usize {
        0
    }
}
impl<T: Float> Eval<T> for ReLu {
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
            .for_each(|(s, i)| *s = i.max(T::zero()));
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
impl<T: Float> BackProp<T> for ReLu {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &mut [T],
        gradient: &mut [T],
    ) {
        debug_assert!(input.len() == self.inputs, "ReLu input");
        debug_assert!(weights.len() == 0, "ReLu, weights");
        debug_assert!(state.len() == self.inputs, "ReLu state");
        debug_assert!(gradient.len() == 0, "ReLu gradient");
        debug_assert!(front.len() == self.inputs, "ReLu front");
        debug_assert!(back.len() == self.inputs, "ReLu back");
        front
            .iter_mut()
            .zip(input.iter())
            .zip(back.iter())
            .for_each(|((f, i), b)| *f = *b * i.signum().max(T::zero()));
    }
}
impl<T: Float> Activation<T> for ReLu {
    fn range(&self) -> (Option<T>, Option<T>) {
        (Some(T::zero()), None)
    }
}
