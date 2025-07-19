use num::Float;

use crate::training::{Activation, BackProp, Eval, Weights};

pub struct Id {
    inputs: usize,
}
impl Id {
    pub fn new(inputs: usize) -> Self {
        Id { inputs }
    }
    pub fn layer<T: Float>(inputs: usize) -> (usize, Box<dyn Activation<T>>) {
        (inputs, Box::new(Id { inputs }))
    }
}
impl<T: Float> Weights<T> for Id {
    fn weights_len(&self) -> usize {
        0
    }
}
impl<T: Float> Eval<T> for Id {
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
            .for_each(|(s, i)| *s = *i);
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
impl<T: Float> BackProp<T> for Id {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &mut [T],
        gradient: &mut [T],
    ) {
        debug_assert!(input.len() == self.inputs, "Id input");
        debug_assert!(weights.len() == 0, "Id, weights");
        debug_assert!(state.len() == self.inputs, "Id state");
        debug_assert!(gradient.len() == 0, "Id gradient");
        debug_assert!(front.len() == self.inputs, "Id front");
        debug_assert!(back.len() == self.inputs, "Id back");
        front
            .iter_mut()
            .zip(input.iter())
            .zip(back.iter())
            .for_each(|((f, _i), b)| *f = *b);
    }
}
impl<T: Float> Activation<T> for Id {
    fn range(&self) -> (Option<T>, Option<T>) {
        (None, None)
    }
}
