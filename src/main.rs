use array_vector_space::ArrayVectorSpace;
use eframe::{CreationContext, egui};
use egui::{
    Color32, Frame, Key, Pos2, Rect, ScrollArea, emath,
    epaint::{self, PathStroke},
    lerp, pos2, remap, vec2,
};
use reinforcement::network::{
    Float, Network,
    activation::{Id, Relu, SigmoidSim},
    layer::Layer,
    layers::Layers,
    reinforcement::{Reinforcement, Reward},
};

const MAX_TICKS: usize = 200;

fn add() {
    //TODO use identity for activation for this case
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
}

fn xor() {
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
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    add();
    xor();

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
    net: Reinforcement<6, 2, Layers<6, 10, Relu, Layers<10, 10, Relu, Layer<10, 2, SigmoidSim>>>>,
    alpha: Float,
    relaxation: Float,
    shape: Float,
    obstacles: [([Float; 2], Float); 4],
    targets: [[Float; 2]; 4],
    rewards: [Reward; 4],
    start: [Float; 2],
    nb_reinforcement: usize,
    ds: [Float; 4],
    dtots: [Float; 4],
    trajectory: [[[Float; 2]; MAX_TICKS]; 4],
}

impl Content {
    fn new() -> Self {
        let targets = [
            [100.0 as Float, 100.0],
            [100.0, -100.0],
            [-100.0, -100.0],
            [-100.0, 100.0],
        ];
        let mut s = Content {
            net: Default::default(),
            alpha: 1e-3,
            relaxation: 1e-3,
            shape: 30.0,
            obstacles: targets.map(|t| (t.scal_mul(0.5), 40.0)),
            targets,
            start: [0.0; 2],
            rewards: [Reward::default(); 4],
            nb_reinforcement: 166,
            ds: [0.0; 4],
            dtots: [0.0; 4],
            trajectory: [[[0.0; 2]; MAX_TICKS]; 4],
        };
        s.net.randomize();
        s
    }
    fn reinforce(&mut self) {
        for _ in 0..self.nb_reinforcement {
            let mut poss = self.targets.map(|_| (self.start, [0.0; 2]));
            self.dtots = self.targets.map(|_| 0.0);
            self.ds = self.targets.map(|_| 0.0);
            for (((((target, (pos, speed)), dtot), d), reward), j) in self
                .targets
                .iter()
                .zip(poss.iter_mut())
                .zip(self.dtots.iter_mut())
                .zip(self.ds.iter_mut())
                .zip(self.rewards.iter_mut())
                .zip(0..)
            {
                let mut ticks = 0;
                self.net.reinforce(self.relaxation, self.alpha, |net| {
                    let input = [pos[0], pos[1], speed[0], speed[1], target[0], target[1]];
                    let output = net.pert_forward(input, self.shape);
                    *speed = speed.add(output.scal_mul(1e-1));
                    let next_pos = pos.add(*speed);
                    if self
                        .obstacles
                        .into_iter()
                        .map(|(p, r)| next_pos.sub(p).norm2() < r * r)
                        .fold(false, |a, b| a || b)
                    {
                        *speed = [0.0; 2];
                    } else {
                        *pos = next_pos;
                    }
                    self.trajectory[j][ticks] = *pos;
                    ticks += 1;
                    *dtot += speed.norm2().sqrt();
                    if ticks >= MAX_TICKS {
                        *d = target.sub(*pos).norm2().sqrt();
                        Some(reward.update(-*dtot / (1.0 + *d) * 1e-3 - *d))
                    } else {
                        None
                    }
                });
            }
        }
    }
}
impl eframe::App for Content {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.reinforce();
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    *self = Self::new();
                }
                ui.label(format!(
                    "{MAX_TICKS:<3}: target dist {:.1} {:?} | tot dist {:.1} {:?}",
                    self.ds.iter().fold(0.0 as Float, |a, v| a + v * v).sqrt(),
                    self.ds.map(|v| v as i64),
                    self.dtots
                        .iter()
                        .fold(0.0 as Float, |a, v| a + v * v)
                        .sqrt(),
                    self.dtots.map(|v| v as i64)
                ));
            });

            Frame::canvas(ui.style()).show(ui, |ui| {
                let desired_size = ui.available_size();
                let ratio = desired_size.x / desired_size.y;
                let (_id, rect) = ui.allocate_space(desired_size);

                let to_screen = emath::RectTransform::from_to(
                    Rect::from_x_y_ranges(-200.0 * ratio..=200.0 * ratio, -200.0..=200.0),
                    rect,
                );
                let colors = [
                    Color32::from_rgb(255, 0, 0),
                    Color32::from_rgb(0, 255, 0),
                    Color32::from_rgb(0, 0, 255),
                    Color32::from_rgb(255, 255, 0),
                ];
                let shapes = self
                    .trajectory
                    .iter()
                    .zip(colors.into_iter())
                    .map(|(t, c)| {
                        epaint::Shape::line(
                            t.map(|[x, y]| to_screen * pos2(x as _, y as _)).to_vec(),
                            PathStroke::new(5.0, c),
                        )
                    })
                    .chain(self.obstacles.into_iter().map(|([x, y], r)| {
                        epaint::Shape::circle_filled(
                            to_screen * pos2(x as _, y as _),
                            to_screen.scale().y * r as f32,
                            Color32::BLACK,
                        )
                    }))
                    .chain(self.targets.into_iter().map(|[x, y]| {
                        epaint::Shape::circle_filled(
                            to_screen * pos2(x as _, y as _),
                            to_screen.scale().y * 5.0,
                            Color32::GRAY,
                        )
                    }));

                ui.painter().extend(shapes);
            });
        });
        ctx.request_repaint();
    }
}
