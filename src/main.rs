use reinforcement::simulation::{acceleration::Acceleration, with_egui};

fn main() {
    with_egui(|| Acceleration::new());
}
