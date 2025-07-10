use eframe::egui;
use egui::{Frame, Rect, emath};
use reinforcement::{
    network::{Float, Network, activation::Id, layer::Layer, layers::Layers},
    simulation::Sim,
};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
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

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Reinforcement",
        native_options,
        Box::new(|_cc| Ok(Box::new(Content::new()))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(Content::new()))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

struct Content {
    alpha: Float,
    alpha_score: Float,
    relaxation: Float,
    sigma: Float,
    movement: bool,
    simulation: Sim,
}

impl Content {
    fn new() -> Self {
        let alpha = 1e-2;
        let alpha_score = 1e-3;
        let relaxation = 1e-4;
        let sigma = 1e1;
        Content {
            alpha,
            alpha_score,
            relaxation,
            sigma,
            movement: false,
            simulation: Sim::new(alpha, alpha_score, relaxation, sigma),
        }
    }
}
impl eframe::App for Content {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.simulation.reinforce();
            let trajectories = self.simulation.simulate(self.movement);
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    self.simulation.reset();
                }
                ui.toggle_value(&mut self.movement, "move");
            });
            ui.add(
                egui::Slider::new(&mut self.alpha, 1e-10..=1e5)
                    .logarithmic(true)
                    .text("alpha"),
            );
            ui.add(
                egui::Slider::new(&mut self.alpha_score, 1e-10..=1e5)
                    .logarithmic(true)
                    .text("alpha_score"),
            );
            ui.add(
                egui::Slider::new(&mut self.relaxation, 1e-5..=1.0)
                    .logarithmic(true)
                    .text("relaxation"),
            );
            ui.add(
                egui::Slider::new(&mut self.sigma, 1e-6..=1e6)
                    .logarithmic(true)
                    .text("sigma"),
            );

            Frame::canvas(ui.style()).show(ui, |ui| {
                let desired_size = ui.available_size();
                let ratio = desired_size.x / desired_size.y;
                let rx = ratio.max(1.0);
                let ry = ratio.recip().max(1.0);
                let (_id, rect) = ui.allocate_space(desired_size);

                let to_screen = emath::RectTransform::from_to(
                    Rect::from_x_y_ranges(-200.0 * rx..=200.0 * rx, -200.0 * ry..=200.0 * ry),
                    rect,
                );

                ui.painter_at(rect)
                    .extend(self.simulation.shapes(to_screen, trajectories));
            });
        });
        ctx.request_repaint();
    }
}
