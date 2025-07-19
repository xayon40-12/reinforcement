use num::Float;

use super::{Activation, BackProp, Eval, Weights, layer_matrix::LayerMatrix};

pub struct MLP<T: Float> {
    layers: Box<[(LayerMatrix<T>, Box<dyn Activation<T>>)]>,
}

impl<T: Float> MLP<T> {
    pub fn new(mut inputs: usize, layers: Vec<(usize, Box<dyn Activation<T>>)>) -> Self {
        MLP {
            layers: layers
                .into_iter()
                .map(|(outputs, activation)| {
                    let layer = (LayerMatrix::new(inputs, outputs), activation);
                    inputs = outputs;
                    layer
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
    pub fn output_range(&self) -> (Option<T>, Option<T>) {
        self.layers[self.layers.len() - 1].1.range()
    }
    pub fn min_back_front_len(&self) -> usize {
        self.layers
            .iter()
            .fold(self.layers[0].0.inputs(), |a, l| a.max(l.0.outputs()))
    }
}

impl<T: Float> Weights<T> for MLP<T> {
    fn weights_len(&self) -> usize {
        self.layers
            .iter()
            .map(|(l, a)| l.weights_len() + a.weights_len())
            .sum()
    }
}

impl<T: Float> Eval<T> for MLP<T> {
    fn state_len(&self) -> usize {
        self.layers
            .iter()
            .map(|(l, a)| l.state_len() + a.state_len())
            .sum()
    }

    fn eval(&self, input: &[T], weights: &[T], state: &mut [T]) {
        debug_assert!(
            input.len() == self.layers[0].0.inputs()
                && weights.len() == self.weights_len()
                && state.len() == self.state_len()
        );
        let mut input = &*input;
        let mut weights = &*weights;
        let mut state = &mut *state;
        for (l, a) in self.layers.iter() {
            l.eval(
                input,
                &weights[0..l.weights_len()],
                &mut state[0..l.state_len()],
            );
            weights = &weights[l.weights_len()..];
            (input, state) = state.split_at_mut(l.state_len());
            a.eval(
                input,
                &weights[0..a.weights_len()],
                &mut state[0..a.state_len()],
            );
            weights = &weights[a.weights_len()..];
            (input, state) = state.split_at_mut(a.state_len());
        }
    }
    fn input_len(&self) -> usize {
        self.layers[0].0.inputs()
    }
    fn output_len(&self) -> usize {
        self.layers[self.layers.len() - 1].0.outputs()
    }
    fn output<'a>(&self, state: &'a [T]) -> &'a [T] {
        debug_assert!(state.len() == self.state_len());
        &state[self.state_len() - self.layers[self.layers.len() - 1].0.state_len()..]
    }
    fn output_mut<'a>(&self, state: &'a mut [T]) -> &'a mut [T] {
        debug_assert!(state.len() == self.state_len());
        &mut state[self.state_len() - self.layers[self.layers.len() - 1].0.state_len()..]
    }
}
impl<T: Float> BackProp<T> for MLP<T> {
    fn back_prop(
        &self,
        input: &[T],
        weights: &[T],
        state: &[T],
        front: &mut [T],
        back: &mut [T],
        gradient: &mut [T],
    ) {
        let min_back_front_len = self.min_back_front_len();
        debug_assert!(input.len() == self.layers[0].0.inputs(), "MLP input");
        debug_assert!(weights.len() == self.weights_len(), "MLP weigths");
        debug_assert!(state.len() == self.state_len(), "MLP state");
        debug_assert!(gradient.len() == self.weights_len(), "MLP gradient");
        debug_assert!(front.len() >= min_back_front_len, "MLP front");
        debug_assert!(back.len() >= min_back_front_len, "MLP back");

        let mut gradients = Vec::with_capacity(self.layers.len());
        let mut weightss = Vec::with_capacity(self.layers.len());
        let mut gradient = &mut *gradient;
        let mut weights = &*weights;
        for len in self.layers.iter().map(|l| l.0.weights_len()) {
            let (a, b) = gradient.split_at_mut(len);
            gradients.push(a);
            gradient = b;
            let (a, b) = weights.split_at(len);
            weightss.push(a);
            weights = b;
        }
        let mut inputs = Vec::with_capacity(self.layers.len() * 2);
        let mut statess = Vec::with_capacity(self.layers.len() * 2);
        inputs.push(input);
        let mut state = &*state;
        for (i, l) in (0..).zip(self.layers.iter()) {
            let (a, rest) = state.split_at(l.0.state_len());
            let (b, rest) = rest.split_at(l.1.state_len());
            inputs.push(a);
            statess.push(a);
            statess.push(b);
            state = rest;
            if i < self.layers.len() - 1 {
                inputs.push(b);
            }
        }
        self.layers
            .iter()
            .zip(inputs.chunks(2))
            .zip(statess.chunks(2))
            .zip(gradients.into_iter())
            .zip(weightss.into_iter())
            .rev()
            .for_each(|((((l, i), s), g), ws)| {
                let il1 = i[1].len();
                let il0 = i[0].len();
                l.1.back_prop(
                    &i[1],
                    &[],
                    &s[1],
                    &mut front[..il1],
                    &mut back[..il1],
                    &mut [],
                );
                l.0.back_prop(&i[0], ws, &s[0], &mut back[..il0], &mut front[..il1], g);
            });
        front.iter_mut().zip(back.iter()).for_each(|(f, b)| *f = *b);
    }
}
