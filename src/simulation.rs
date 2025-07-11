use std::{iter::once, ops::RangeInclusive, usize};

use egui::{Frame, Rect, Shape, emath::RectTransform};
use futures::channel::mpsc::{Receiver, Sender, channel};

use crate::network::{
    Float,
    reinforcement::{MetaParameters, Reinforce},
};

pub mod acceleration;

pub trait Tag: Copy + Send + 'static {
    fn str(&self) -> &'static str;
}

pub enum Request<T> {
    UpdateParameter(UpadeParameter<T>),
    Reinforce,
    Reset,
    Render(RectTransform),
}

pub enum Reply {
    Shapes(Vec<Shape>),
}

pub enum Parameter<T> {
    Slider {
        tag: T,
        value: Float,
        logarithmic: bool,
        range: RangeInclusive<Float>,
    },
    Toggle {
        tag: T,
        enable: bool,
    },
    Button {
        tag: T,
    },
}

pub enum UpadeParameter<T> {
    Slider { tag: T, value: Float },
    Toggle { tag: T, enable: bool },
    Button { tag: T },
}

pub trait Simulation: Send + 'static {
    type Tag: Tag;
    type Ctx: Clone;
    type ActionIn;
    type ActionOut;
    type PhysicsInfo;

    fn reset(&mut self);

    fn meta_parameters(&self) -> MetaParameters;
    fn egui_parameters(&self) -> Vec<Parameter<Self::Tag>>;
    fn update_parameter(&mut self, update: UpadeParameter<Self::Tag>);

    fn ctx_list(&self) -> Vec<Self::Ctx>;
    fn max_physics_iteration() -> usize;

    fn score_goal(ctx: &Self::Ctx, info: Self::PhysicsInfo) -> (Float, bool);
    fn physics(ctx: &mut Self::Ctx, action: Self::ActionOut) -> Self::PhysicsInfo;
    fn ctx_to_action(ctx: &Self::Ctx) -> Self::ActionIn;

    fn reinforcement_network(
        &mut self,
    ) -> &mut impl Reinforce<ActionIn = Self::ActionIn, ActionOut = Self::ActionOut>;

    fn render_rect(&self) -> Rect;
    fn render(&self, to_screen: RectTransform, trajectories: Vec<Vec<Self::Ctx>>) -> Vec<Shape>;

    fn handle_request(&mut self, request: Request<Self::Tag>) -> Option<Reply> {
        match request {
            Request::UpdateParameter(u) => {
                self.update_parameter(u);
            }
            Request::Reinforce => self.reinforce(),
            Request::Reset => self.reset(),
            Request::Render(to_screen) => {
                let trajectories = self.simulate();
                return Some(Reply::Shapes(self.render(to_screen, trajectories)));
            }
        }
        None
    }
    fn reinforce(&mut self) {
        let meta_parameters = self.meta_parameters();
        let mut ctx_list = self.ctx_list();
        self.reinforcement_network().reinforce(
            meta_parameters,
            &mut ctx_list,
            Self::ctx_to_action,
            Self::max_physics_iteration(),
            |ctx: &mut Self::Ctx, action: Self::ActionOut| {
                let info = Self::physics(ctx, action);
                Self::score_goal(ctx, info)
            },
        );
    }
    fn simulate(&mut self) -> Vec<Vec<Self::Ctx>> {
        self.ctx_list()
            .into_iter()
            .map(|mut ctx| {
                once(ctx.clone())
                    .chain((0..Self::max_physics_iteration()).map_while(|_| {
                        let action_in = self
                            .reinforcement_network()
                            .forward(Self::ctx_to_action(&ctx));
                        let info = Self::physics(&mut ctx, action_in);
                        let done = Self::score_goal(&ctx, info).1;
                        if done { None } else { Some(ctx.clone()) }
                    }))
                    .collect()
            })
            .collect()
    }
}

pub struct SimulationGUI<S: Simulation, FS: Fn() -> S> {
    render_rect: Rect,
    parameters: Vec<Parameter<S::Tag>>,
    simulation_request_tx: Sender<Request<S::Tag>>,
    simulation_reply_rx: Receiver<Reply>,
    create_simulation: FS,
    cached_shapes: Vec<Shape>,
}

fn handler<S: Simulation>(mut simulation: S) -> (Sender<Request<S::Tag>>, Receiver<Reply>) {
    let (request_tx, mut request_rx) = channel::<Request<S::Tag>>(100);
    let (mut reply_tx, reply_rx) = channel::<Reply>(100);
    let mut main_logic = move || {
        let request = match request_rx.try_next() {
            Ok(Some(request)) => request,
            Ok(None) => {
                return false;
            }
            Err(_) => Request::Reinforce,
        };
        if let Some(reply) = simulation.handle_request(request) {
            if let Err(err) = reply_tx.try_send(reply) {
                log::log!(log::Level::Error, "{err}");
                return false;
            }
        }
        true
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || while main_logic() {});
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        loop {
            for _ in 0..100 {
                //FIXME: this is to have better performence, but this 100 might be too large for slower simulations
                main_logic();
            }
            gloo_timers::future::TimeoutFuture::new(0).await;
        }
    });
    (request_tx, reply_rx)
}

impl<S: Simulation, FS: Fn() -> S> SimulationGUI<S, FS> {
    pub fn new(create_simulation: FS) -> Self {
        let simulation = create_simulation();
        let render_rect = simulation.render_rect();
        let parameters = simulation.egui_parameters();
        let (simulation_request_tx, simulation_reply_rx) = handler(simulation);
        SimulationGUI {
            render_rect,
            parameters,
            simulation_request_tx,
            simulation_reply_rx,
            create_simulation,
            cached_shapes: Vec::new(),
        }
    }
}
impl<S: Simulation, FS: Fn() -> S> eframe::App for SimulationGUI<S, FS> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut requests = vec![];
        if let Ok(Some(reply)) = self.simulation_reply_rx.try_next() {
            match reply {
                Reply::Shapes(shapes) => self.cached_shapes = shapes,
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("reset").clicked() {
                    requests.push(Request::Reset);
                }
            });
            for p in self.parameters.iter_mut() {
                match p {
                    Parameter::Slider {
                        tag,
                        value,
                        logarithmic,
                        range,
                    } => {
                        if ui
                            .add(
                                egui::Slider::new(value, range.clone())
                                    .logarithmic(*logarithmic)
                                    .text(tag.str()),
                            )
                            .changed()
                        {
                            requests.push(Request::UpdateParameter(UpadeParameter::Slider {
                                tag: tag.clone(),
                                value: *value,
                            }));
                        }
                    }
                    Parameter::Toggle { tag, enable } => {
                        if ui.toggle_value(enable, tag.str()).changed() {
                            requests.push(Request::UpdateParameter(UpadeParameter::Toggle {
                                tag: tag.clone(),
                                enable: *enable,
                            }))
                        }
                    }
                    Parameter::Button { tag } => {
                        if ui.button(tag.str()).clicked() {
                            requests.push(Request::UpdateParameter(UpadeParameter::Button {
                                tag: tag.clone(),
                            }))
                        }
                    }
                }
            }

            Frame::canvas(ui.style()).show(ui, |ui| {
                let desired_size = ui.available_size();
                let ratio = desired_size.x / desired_size.y;
                let rx = ratio.max(1.0);
                let ry = ratio.recip().max(1.0);
                let (_id, rect) = ui.allocate_space(desired_size);

                let to_screen = RectTransform::from_to(
                    Rect::from_x_y_ranges(
                        self.render_rect.min.x * rx..=self.render_rect.max.x * rx,
                        self.render_rect.min.y * ry..=self.render_rect.max.y * ry,
                    ),
                    rect,
                );
                requests.push(Request::Render(to_screen));

                ui.painter_at(rect).extend(self.cached_shapes.clone());
            });
        });
        for request in requests {
            if let Err(err) = self.simulation_request_tx.try_send(request) {
                log::log!(log::Level::Error, "{err}");
                let simulation = (self.create_simulation)();
                let (tx, rx) = handler(simulation);
                self.simulation_request_tx = tx;
                self.simulation_reply_rx = rx;
            }
        }
        ctx.request_repaint();
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn with_egui<S: Simulation>(create_simulation: impl Fn() -> S) {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions::default();
    if let Err(err) = eframe::run_native(
        "Reinforcement",
        native_options,
        Box::new(|_cc| Ok(Box::new(SimulationGUI::new(create_simulation)))),
    ) {
        log::log!(log::Level::Error, "{err}");
    }
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
pub fn with_egui<S: Simulation>(create_simulation: impl Fn() -> S) {
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
                Box::new(|_cc| Ok(Box::new(SimulationGUI::new(|| Sim::new())))),
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
