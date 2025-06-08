use reinforcement::network::{Layer, Layers, Network};

fn main() {
    let mut net: Layers<2, 3, 1, Layers<3, 3, 1, Layer<3, 1>>> = Default::default();
    net.train(
        10000,
        1e-2,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [0.0]),
            ([1.0, 0.0], [1.0]),
        ],
    );
}
