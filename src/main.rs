use std::process::exit;

use reinforcement::simulation::{acceleration::Acceleration, with_egui};

fn main() {
    if cfg!(debug_assertions) {
        use reinforcement::training::{
            activations::{id::Id, tanh::Tanh},
            mlp::MLP,
            tests::test_value_adam,
            trainer::Trainer,
        };
        let mut net: Trainer<f64> = Trainer::new(MLP::new(2, vec![Tanh::layer(3), Id::layer(1)]));
        net.randomize_weights(1.0);
        net.train(
            1001,
            1e-1,
            &[
                (&[0.0, 0.0], &[0.0]),
                (&[0.0, 1.0], &[1.0]),
                (&[1.0, 1.0], &[2.0]),
                (&[1.0, 0.0], &[1.0]),
                (&[-1.0, -6.0], &[-7.0]),
                (&[-1.0, 6.0], &[5.0]),
            ],
        );
        test_value_adam();

        exit(0);
    }
    with_egui(|| Acceleration::new());
}
