use num::Float;
use rand::seq::SliceRandom;

pub mod activations;
pub mod layer_matrix;
pub mod least_squar_value;
pub mod mlp;
pub mod optimizers;
pub mod policies;
pub mod tests;
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
// NOTE: It might be more efficient to add to the existing gradient in back_prop and compute gradient
pub trait BackProp<T: Float>: Eval<T> {
    /// This function is supposed to completely overwrite the gradient, not add to it
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
pub trait Gradient<T: Float>: Eval<T> {
    /// This function is supposed to completely overwrite the gradient, not add to it.
    /// Also, it is not supposed to call eval, but instead call output.
    fn compute_gradient(&self, input: &[T], weights: &[T], state: &mut [T], gradient: &mut [T]);
}
pub trait Policy<T: Float> {
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
}
/// It is `Gradient` with respect to the scalar probability
pub trait StochasticPolicy<T: Float>: Gradient<T> {
    fn probability(&self, state: &[T]) -> T;
    fn stochastic_eval(&self, input: &[T], weights: &[T], state: &mut [T]);
    fn stochastic_output<'a>(&self, state: &'a [T]) -> &'a [T];
}

pub trait Value<T: Float>: Gradient<T> {
    fn set_target(&mut self, target: T, state: &mut [T]);
    fn target(&self, state: &[T]) -> T;
    fn value(&self, state: &[T]) -> T;
}

pub trait Activation<T: Float>: BackProp<T> {
    fn range(&self) -> (Option<T>, Option<T>);
}

pub struct TimeStep<'a, T: Float> {
    pub input: &'a [T],
    pub state: &'a mut [T],
}

pub trait Optimizer<T: Float> {
    /// This function is supposed to perform gradient descent
    fn step(&mut self, weights: &mut [T], gradient: &mut [T]);
    fn optimize<G: Gradient<T>>(
        &mut self,
        epochs: usize,
        minibatch_size: usize,
        to_optimize: &mut G,
        weights: &mut [T],
        gradient: &mut [T],
        tmp_gradient: &mut [T],
        time_steps: &mut [TimeStep<T>],
        descent: bool,
    ) {
        debug_assert!(minibatch_size > 0, "minibatch size must be positive");
        let ascent_sign = if descent { T::one() } else { -T::one() };

        let mut rng = rand::rng();

        let recip_minibatch_size = T::from(minibatch_size).unwrap().recip();
        let normalization = ascent_sign * recip_minibatch_size;
        for _ in 0..epochs {
            time_steps.shuffle(&mut rng); // FIXME: it might be more efficient to have an array of indices and shuffle that array instead of the array of time_steps. However, this array would have to be provided as &mut [usize] in order to avoid allocation in this part for later GPU switching 
            for minibatch in time_steps.chunks_exact_mut(minibatch_size) {
                gradient.iter_mut().for_each(|g| *g = T::zero());
                for TimeStep { input, state } in minibatch {
                    to_optimize.eval(input, weights, state);
                    to_optimize.compute_gradient(input, weights, state, tmp_gradient);
                    gradient
                        .iter_mut()
                        .zip(tmp_gradient.iter())
                        .for_each(|(g, tmp_g)| *g = *g + *tmp_g);
                }
                gradient.iter_mut().for_each(|g| *g = *g * normalization);
                self.step(weights, gradient);
            }
        }
    }
}
