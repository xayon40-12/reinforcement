use array_vector_space::ArrayVectorSpace;
use eframe::egui;
use egui::{
    Color32, Frame, Rect, Stroke, emath,
    epaint::{self, PathStroke},
    pos2,
};
use reinforcement::network::{
    Float, ForwardNetwork, Network,
    activation::{Id, Relu, SigmoidSim},
    layer::Layer,
    layers::Layers,
    reinforcement::{Reinforcement, Reward},
};

const MAX_TICKS: usize = 100;

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
    net: Reinforcement<
        6,
        2,
        // Layers<6, 10, Relu, Layers<10, 10, Relu, Layer<10, 2, SigmoidSim>>>,
        Layers<6, 10, Relu, Layers<10, 10, Relu, Layers<10, 10, Relu, Layer<10, 2, SigmoidSim>>>>,
    >,
    alpha: Float,
    relaxation: Float,
    shape: Float,
    obstacles: [([Float; 2], Float); 44],
    targets: [([Float; 2], Float); 4],
    rewards: [Reward; 4],
    starts: [[Float; 2]; 4],
    poss: [[Float; 2]; 4],
    movement: bool,
    learn: bool,
    nb_reinforcement: usize,
    trajectory: [[[Float; 2]; MAX_TICKS]; 4],
}

impl Content {
    fn new() -> Self {
        let targets = [
            ([100.0 as Float, 100.0], 5.0),
            ([100.0, -100.0], 5.0),
            ([-100.0, -100.0], 5.0),
            ([-100.0, 100.0], 5.0),
        ];
        let starts = std::array::from_fn(|i| {
            targets[(i + 1) % 4]
                .0
                .add(targets[(i + 2) % 4].0)
                .scal_mul(0.25)
        });
        let obstacles: [_; 44] = std::array::from_fn(|i| {
            if i < 4 {
                (targets[i].0.scal_mul(0.5), 40.0)
            } else {
                let a = (i - 4) as Float / 40.0 * 3.1415 * 2.0;
                let r = 200.0;
                ([a.cos() * r, a.sin() * r], 30.0)
            }
        });
        let mut s = Content {
            net: Default::default(),
            alpha: 1e-4,
            relaxation: 1e-3,
            shape: 1e3,
            obstacles,
            targets,
            starts,
            poss: starts,
            movement: true,
            learn: true,
            rewards: [Reward::default(); 4],
            nb_reinforcement: 1,
            trajectory: [[[0.0; 2]; MAX_TICKS]; 4],
        };
        s.net.randomize();
        s
    }
    fn reset(&mut self) {
        self.net.randomize();
        self.rewards = [Reward::default(); 4];
        self.poss = self.starts;
    }
    fn reinforce(&mut self) {
        for _ in 0..self.nb_reinforcement {
            type Ctx = (
                Reward,
                [[f64; 2]; 100],
                ([f64; 2], f64),
                [f64; 2],
                [f64; 2],
                f64,
                f64,
                f64,
                usize,
            );
            let mut ctxs: [Ctx; 4] = std::array::from_fn(|i| {
                (
                    self.rewards[i],
                    self.trajectory[i],
                    self.targets[i],
                    self.poss[i],
                    [0.0; 2],
                    0.0,
                    0.0,
                    0.0,
                    0,
                )
            });
            let sim =
                |(reward, trajectory, (target, _), pos, speed, d, dtot, hit, ticks): &mut Ctx,
                 output: [Float; 2]| {
                    *speed = speed.add(output.scal_mul(1e-1));
                    let next_pos = pos.add(*speed);
                    let mut collision = false;
                    for (o, r) in self.obstacles {
                        if next_pos.sub(o).norm2() < r * r {
                            *speed = [0.0; 2];
                            collision = true;
                        }
                        let d = pos.sub(o).norm2().sqrt() - r;
                        *hit += 1e3 / (1.0 + d).powi(3);
                    }
                    if !collision {
                        *pos = next_pos;
                    }
                    trajectory[*ticks] = *pos;
                    *ticks += 1;

                    for (o, r) in self.obstacles {
                        let d = pos.sub(o).norm2().sqrt() - r;
                        *hit += 1e3 / (1.0 + d).powi(3);
                    }
                    *dtot += speed.norm2().sqrt();
                    if *ticks >= MAX_TICKS {
                        *d = target.sub(*pos).norm2().sqrt();
                        // Some(reward.update(-*dtot / (1.0 + *d).powi(4) - *d - *hit))
                        Some(reward.update(-*d - *hit * 1e-2))
                    } else {
                        None
                    }
                };
            if self.learn {
                self.net
                    .reinforce(self.relaxation, self.alpha, &mut ctxs, |ctx, net| {
                        let output = {
                            let (_, _, (target, _), pos, speed, ..) = ctx;
                            net.pert_forward(
                                [pos[0], pos[1], speed[0], speed[1], target[0], target[1]],
                                self.shape,
                            )
                        };
                        sim(ctx, output)
                    });
            } else {
                for ctx in ctxs.iter_mut() {
                    loop {
                        let output = {
                            let (_, _, (target, _), pos, speed, ..) = ctx;
                            self.net
                                .forward([pos[0], pos[1], speed[0], speed[1], target[0], target[1]])
                        };
                        if sim(ctx, output).is_some() {
                            break;
                        }
                    }
                }
            }
            for (i, (reward, trajectory, (t, r), ..)) in (0..).zip(ctxs.into_iter()) {
                self.rewards[i] = reward;
                self.trajectory[i] = trajectory;
                let next = trajectory[0];
                if next.sub(t).norm2() < r * r {
                    self.poss[i] = self.starts[i];
                } else if self.movement {
                    self.poss[i] = next;
                }
            }
        }
    }
}
impl eframe::App for Content {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.stable_dt) < 0.0166 {
            self.nb_reinforcement = 1 + self.nb_reinforcement * 10 / 9;
        } else {
            self.nb_reinforcement = 1 + self.nb_reinforcement * 9 / 10;
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            self.reinforce();
            ui.label(format!("{}", self.nb_reinforcement));
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    self.reset();
                }
                if ui.toggle_value(&mut self.movement, "move").changed() {
                    if !self.movement {
                        self.poss = self.starts;
                    }
                }
                if ui.toggle_value(&mut self.learn, "learn").changed() {
                    self.nb_reinforcement = 1;
                    self.poss = self.starts;
                }
            });
            ui.add(
                egui::Slider::new(&mut self.alpha, 1e-10..=1e5)
                    .logarithmic(true)
                    .text("alpha"),
            );
            ui.add(
                egui::Slider::new(&mut self.relaxation, 1e-5..=1.0)
                    .logarithmic(true)
                    .text("relaxation"),
            );
            ui.add(
                egui::Slider::new(&mut self.shape, 1.0..=1e6)
                    .logarithmic(true)
                    .text("shape"),
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
                let colors = [
                    Color32::from_rgb(255, 0, 0),
                    Color32::from_rgb(0, 255, 0),
                    Color32::from_rgb(0, 0, 255),
                    Color32::from_rgb(255, 192, 0),
                ];
                let shapes =
                    self.trajectory
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
                                Color32::GRAY,
                            )
                        }))
                        .chain(self.targets.into_iter().zip(colors.into_iter()).map(
                            |(([x, y], r), c)| {
                                epaint::Shape::circle_stroke(
                                    to_screen * pos2(x as _, y as _),
                                    to_screen.scale().y * r as f32,
                                    Stroke::new(1.0, c),
                                )
                            },
                        ))
                        .chain(self.starts.into_iter().zip(colors.into_iter()).map(
                            |([x, y], c)| {
                                epaint::Shape::circle_filled(
                                    to_screen * pos2(x as _, y as _),
                                    to_screen.scale().y * 5.0,
                                    c,
                                )
                            },
                        ))
                        .chain(
                            self.trajectory
                                .into_iter()
                                .map(|t| t[0])
                                .zip(colors.into_iter())
                                .map(|([x, y], c)| {
                                    epaint::Shape::circle_filled(
                                        to_screen * pos2(x as _, y as _),
                                        to_screen.scale().y * 3.0,
                                        c,
                                    )
                                }),
                        );

                ui.painter().extend(shapes);
            });
        });
        ctx.request_repaint();
    }
}
