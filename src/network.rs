use activation::Activation;

pub mod activation;
pub mod layer;
pub mod layers;
pub mod reinforcement;

pub type Float = f64;

pub trait JoinNetwork<const NI: usize> {
    type OutA: Activation;
}
pub trait ForwardNetwork<const NI: usize, const NO: usize>: JoinNetwork<NI> {
    fn forward(&mut self, input: [Float; NI]) -> [Float; NO];
}
pub trait Network<const NI: usize, const NO: usize>: ForwardNetwork<NI, NO> + Clone {
    fn randomize(&mut self);
    /// The parameter $r$ is the ratio $0 < r <= 1$ for the relaxation of the update of the gradient. A value of $r = 1$ correspond to keep only the new value of the gradient, whereas $r = 0.5$ will average the new and last value.
    fn update_gradient(&mut self, relaxation: Float, delta: [Float; NO]) -> [Float; NI];
    fn reset_gradient(&mut self);
    fn apply_gradient(&mut self, alpha: Float);
    fn rescale_gradient(&mut self, a: Float);
    fn norm2_gradient(&self) -> Float;
    fn normalize_gradient(&mut self) {
        self.rescale_gradient(self.norm2_gradient().sqrt().recip());
    }
    fn add_gradient(&mut self, rhs: &Self);

    // use reinforcement::network::{Network, activation::Id, layer::Layer, layers::Layers};
    // let mut net: Layers<2, 3, Id, Layer<3, 1, Id>> = Default::default();
    //  net.train(
    //      1000,
    //      1e-2,
    //      &[
    //          ([0.0, 0.0], [0.0]),
    //          ([0.0, 1.0], [1.0]),
    //          ([1.0, 1.0], [2.0]),
    //          ([1.0, 0.0], [1.0]),
    //          ([-1.0, -6.0], [-7.0]),
    //          ([-1.0, 6.0], [5.0]),
    //      ],
    //  );
    //  let mut net: Layers<2, 2, Id, Layer<2, 1, Id>> = Default::default();
    //  net.train(
    //      100,
    //      1e-1,
    //      &[
    //          ([0.0, 0.0], [0.0]),
    //          ([0.0, 1.0], [1.0]),
    //          ([1.0, 1.0], [0.0]),
    //          ([1.0, 0.0], [1.0]),
    //      ],
    //  );
    fn train(
        &mut self,
        iterations: usize,
        alpha: Float,
        training_data: &[([Float; NI], [Float; NO])],
    ) {
        let align = iterations.ilog10() as usize;
        for i in 0..iterations {
            let mut error = 0.0;
            for &(inputs, targets) in training_data.iter() {
                let outputs = self.forward(inputs);
                if i % (iterations / 10) == 0 {
                    print!("{outputs:?} ");
                }
                let delta = std::array::from_fn(|i| (targets[i] - outputs[i]));
                error += delta.iter().map(|d| d * d).sum::<Float>();
                self.update_gradient(1.0, delta);
                self.normalize_gradient();
                self.apply_gradient(alpha);
            }
            if i % (iterations / 10) == 0 {
                println!("\n{i:<align$}: error = {:.1e}", error.sqrt());
            }
        }
    }
    fn output_ranges(&self) -> [(Option<Float>, Option<Float>); NO];
}
