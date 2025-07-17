use reinforcement::{
    network::activation::Relu,
    simulation::{acceleration::Acceleration, with_egui},
    weights::{MLP, ReLu, Trainer},
};

fn main() {
    use reinforcement::network::{Network, activation::Id, layer::Layer, layers::Layers};
    let mut net: Layers<2, 3, Id, Layer<3, 1, Id>> = Default::default();
    net.train(
        1000,
        1e-2,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [2.0]),
            ([1.0, 0.0], [1.0]),
            ([-1.0, -6.0], [-7.0]),
            ([-1.0, 6.0], [5.0]),
        ],
    );

    let mut net: Trainer<f64> = Trainer::new(MLP::new(2, vec![ReLu::layer(3), ReLu::layer(1)]));
    net.randomize_weights(0.1);
    net.train(
        10,
        1e-2,
        &[
            (&[0.0, 0.0], &[0.0]),
            (&[0.0, 1.0], &[1.0]),
            (&[1.0, 1.0], &[2.0]),
            (&[1.0, 0.0], &[1.0]),
            (&[-1.0, -6.0], &[-7.0]),
            (&[-1.0, 6.0], &[5.0]),
        ],
    );

    let mut net: Layers<2, 2, Id, Layer<2, 1, Id>> = Default::default();
    net.train(
        100,
        1e-1,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [0.0]),
            ([1.0, 0.0], [1.0]),
        ],
    );
    // with_egui(|| Acceleration::new());
}
