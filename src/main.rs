use reinforcement::network::{Layer, Layers, Network};

fn add() {
    let mut net: Layers<2, 3, 1, Layer<3, 1>> = Default::default();
    net.train(
        10000,
        1e-3,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [2.0]),
            ([1.0, 0.0], [1.0]),
            ([-1.0, -6.0], [-7.0]),
            ([-1.0, 6.0], [5.0]),
        ],
    );
}

fn xor() {
    let mut net: Layers<2, 2, 1, Layers<2, 2, 1, Layer<2, 1>>> = Default::default();
    net.train(
        100000,
        1e-2,
        &[
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [0.0]),
            ([1.0, 0.0], [1.0]),
        ],
    );
}

fn main() {
    add();
    xor();
}
