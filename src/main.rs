use array_vector_space::ArrayVectorSpace;
use boxarray::boxarray_;
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
    reinforcement::{Reinforcement, Score},
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
    net: Reinforcement<6, 2, Layers<6, 16, Relu, Layers<16, 16, Relu, Layer<16, 2, SigmoidSim>>>>,
    alpha: Float,
    relaxation: Float,
    sigma: Float,
    obstacles: [([Float; 2], Float); 44],
    targets: [([Float; 2], Float); 4],
    scores: [Score; 4],
    starts: [[Float; 2]; 4],
    poss: [[Float; 2]; 4],
    speeds: [[Float; 2]; 4],
    movement: bool,
    learn: bool,
    nb_reinforcement: usize,
    tot_reinforcement: usize,
    trajectory: [[([Float; 2], [Float; 2]); MAX_TICKS]; 4],
    traj_ticks: [usize; 4],
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
            relaxation: 1e-4,
            sigma: 1e-2,
            obstacles,
            targets,
            starts,
            poss: starts,
            speeds: [[0.0; 2]; 4],
            movement: false,
            learn: true,
            scores: [Score::default(); 4],
            nb_reinforcement: 1,
            tot_reinforcement: 0,
            trajectory: [[([0.0; 2], [0.0; 2]); MAX_TICKS]; 4],
            traj_ticks: [0; 4],
        };
        s.net.randomize();
        s
    }
    fn reset(&mut self) {
        self.tot_reinforcement = 0;
        self.net.randomize();
        self.scores = [Score::default(); 4];
        self.poss = self.starts;
        self.speeds = [[0.0; 2]; 4];
    }
    fn reinforce(&mut self) {
        for _ in 0..self.nb_reinforcement {
            type Ctx = (
                [([Float; 2], [Float; 2]); MAX_TICKS],
                ([Float; 2], Float),
                [Float; 2],
                [Float; 2],
                usize,
            );
            let mut ctxs: Box<[(Ctx, Score); 4]> = boxarray_(|((), i)| {
                (
                    (
                        self.trajectory[i],
                        self.targets[i],
                        self.poss[i],
                        self.speeds[i],
                        0,
                    ),
                    self.scores[i],
                )
            });
            let sim = |(trajectory, (target, r), pos, speed, ticks): &mut Ctx,
                       output: [Float; 2]| {
                let mut score = 0.0;
                *speed = speed.add(output.scal_mul(1e-1));
                let next_pos = pos.add(*speed);
                let mut collision = false;
                for (o, r) in self.obstacles {
                    if next_pos.sub(o).norm2() < r * r {
                        *speed = [0.0; 2];
                        collision = true;
                    }
                }
                if !collision {
                    *pos = next_pos;
                }
                trajectory[*ticks] = (*pos, *speed);
                *ticks += 1;

                for (o, r) in self.obstacles {
                    let d = pos.sub(o).norm2().sqrt() - r;
                    score -= 1e1 / (1.0 + d).powi(3);
                    if next_pos.sub(o).norm2() < r * r {
                        score -= 1e2 * speed.norm2();
                    }
                }
                score -= speed.norm2().sqrt();
                let d = target.sub(*pos).norm2().sqrt();
                score -= d;
                if d < *r {
                    score += 1e2;
                }
                (score, *ticks >= MAX_TICKS || d < *r)
            };
            let ctx_to_net = |ctx: &Ctx| {
                let (_, (target, _), pos, speed, ..) = ctx;
                [pos[0], pos[1], speed[0], speed[1], target[0], target[1]]
            };
            if self.learn {
                self.net.reinforce(
                    self.relaxation,
                    self.alpha,
                    self.sigma,
                    &mut ctxs,
                    ctx_to_net,
                    sim,
                );
            } else {
                for (ctx, _) in ctxs.iter_mut() {
                    loop {
                        if sim(ctx, self.net.forward(ctx_to_net(ctx))).1 {
                            break;
                        }
                    }
                }
            }
            for (i, ((trajectory, (t, r), .., ticks), score)) in (0..).zip(ctxs.into_iter()) {
                self.scores[i] = score;
                self.trajectory[i] = trajectory;
                self.traj_ticks[i] = ticks;
                let (n_pos, n_speed) = trajectory[0];
                if n_pos.sub(t).norm2() < r * r {
                    self.poss[i] = self.starts[i];
                    self.speeds[i] = [0.0; 2];
                } else if self.movement {
                    self.poss[i] = n_pos;
                    self.speeds[i] = n_speed;
                }
            }
        }
        self.tot_reinforcement += self.nb_reinforcement;
    }
}
impl eframe::App for Content {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // let time = ctx.input(|i| i.time) as Float;
        // let r = 100.0 * (2.0 as Float).sqrt();
        // for (t, i) in self.targets.iter_mut().zip(0..) {
        //     let theta = ((time * 1e1 as Float).floor() + i as Float) * 0.25 * 2.0 * 3.1415;
        //     *t = ([theta.cos() * r, theta.sin() * r], t.1);
        // }
        if ctx.input(|i| i.stable_dt) < 0.0166 {
            self.nb_reinforcement = 1 + self.nb_reinforcement * 10 / 9;
        } else {
            self.nb_reinforcement = 1 + self.nb_reinforcement * 9 / 10;
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            self.reinforce();
            ui.label(format!("{}", self.nb_reinforcement));
            ui.label(format!("{}", self.tot_reinforcement.ilog10()));
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    self.reset();
                }
                if ui.toggle_value(&mut self.movement, "move").changed() {
                    if !self.movement {
                        self.poss = self.starts;
                        self.speeds = [[0.0; 2]; 4];
                    }
                }
                if ui.toggle_value(&mut self.learn, "learn").changed() {
                    self.nb_reinforcement = 1;
                    self.poss = self.starts;
                    self.speeds = [[0.0; 2]; 4];
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
                egui::Slider::new(&mut self.sigma, 1e-6..=1e0)
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
                let colors = [
                    Color32::from_rgb(255, 0, 0),
                    Color32::from_rgb(0, 255, 0),
                    Color32::from_rgb(0, 0, 255),
                    Color32::from_rgb(255, 192, 0),
                ];
                let sigmas =
                    self.trajectory
                        .iter()
                        .zip(self.traj_ticks.iter())
                        .zip(colors.into_iter())
                        .map(|((t, n), c)| {
                            epaint::Shape::line(
                                t.iter()
                                    .take(*n)
                                    .map(|&([x, y], _)| to_screen * pos2(x as _, y as _))
                                    .collect(),
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
                                .map(|(([x, y], _), c)| {
                                    epaint::Shape::circle_filled(
                                        to_screen * pos2(x as _, y as _),
                                        to_screen.scale().y * 3.0,
                                        c,
                                    )
                                }),
                        );

                ui.painter().extend(sigmas);
            });
        });
        ctx.request_repaint();
    }
}
