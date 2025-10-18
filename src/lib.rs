use std::{collections::HashMap, sync::Arc, sync::Mutex};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey, SmolStr},
    window::Window,
};
use instant::Instant;
use glam::Vec2;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use once_cell::sync::OnceCell;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{future_to_promise}; // Import future_to_promise
#[cfg(target_arch = "wasm32")]
use js_sys::Promise;

mod models;
mod camera;
mod color;
mod scene;
mod ui_events;
mod app_state;

use ui_events::UserCommand;
use app_state::State;
#[cfg(target_arch = "wasm32")]
use scene::network::FullTopologyData;

#[cfg(target_arch = "wasm32")]
static WASM_API_INSTANCE: OnceCell<WasmApi> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
static WASM_READY_FLUME_CHANNEL: OnceCell<(flume::Sender<()>, flume::Receiver<()>)> = OnceCell::new();


struct App {
    window: Option<Arc<Window>>,
    state: Arc<Mutex<Option<State>>>, // Wrapped in Arc<Mutex> for interior mutability and potential Send (if State itself were Send)
    #[cfg(target_arch = "wasm32")]
    proxy: Option<EventLoopProxy<UserCommand>>,
}

impl App {
    fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<UserCommand>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let app_proxy = event_loop.create_proxy();

        #[cfg(target_arch = "wasm32")]
        {
            let wasm_api_instance = WasmApi { proxy: app_proxy.clone() };
            if WASM_API_INSTANCE.set(wasm_api_instance).is_err() {
                log::warn!("WASM_API_INSTANCE was already set. This should only happen once.");
            }
        }

        Self {
            window: None,
            state: Arc::new(Mutex::new(None)),
            #[cfg(target_arch = "wasm32")]
            proxy: Some(app_proxy),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn get_window_size(&self) -> Option<winit::dpi::PhysicalSize<u32>> {
        self.window.as_ref().map(|w| w.inner_size())
    }
}

impl ApplicationHandler<UserCommand> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes()
            .with_title("WDMView Graph Topology");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut state = pollster::block_on(State::new(window)).unwrap();
            let current_size = self.get_window_size().unwrap();
            state.resize(current_size.width, current_size.height);
            self.state.lock().unwrap().replace(state); // Set state within the Mutex
            // Request redraw using App's window handle
            self.window.as_ref().unwrap().request_redraw();
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Clone Arc<Mutex<Option<State>>> and Arc<Window> for the async task
            let state_arc_for_spawn = self.state.clone();
            let window_for_state_new = window.clone(); // Pass clone to State::new
            let proxy_for_init_notification = self.proxy.as_ref().expect("App proxy not set").clone();

            wasm_bindgen_futures::spawn_local(async move {
                match State::new(window_for_state_new.clone()).await { // Use clone for State::new
                    Ok(mut state_instance) => {
                        log::info!("WASM State created in async task.");
                        let initial_size = window_for_state_new.inner_size();
                        state_instance.resize(initial_size.width, initial_size.height);

                        {
                            let mut app_state_guard = state_arc_for_spawn.lock().unwrap();
                            app_state_guard.replace(state_instance);
                        }
                        log::info!("WASM State assigned to App. Sending initialization notification.");
                        if proxy_for_init_notification.send_event(UserCommand::StateInitialized).is_err() {
                            log::error!("Failed to send StateInitialized event.");
                        }
                    },
                    Err(e) => log::error!("Failed to create State in WASM: {:?}", e),
                }
            });
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserCommand) {
        match event {
            UserCommand::StateInitialized => {
                log::info!("WASM State initialized and ready.");
                // Signal to the promise resolver
                #[cfg(target_arch = "wasm32")]
                if let Some((sender, _)) = WASM_READY_FLUME_CHANNEL.get() {
                    if let Err(e) = sender.send(()) {
                        log::error!("Failed to send WASM ready signal: {:?}", e);
                    }
                }
                if let Some(w_handle) = self.window.as_ref() {
                    w_handle.request_redraw();
                }
            }
            _ => {
                if let Some(state) = &mut *self.state.lock().unwrap() {
                    state.process_command(event);
                    if let Some(w_handle) = self.window.as_ref() {
                        w_handle.request_redraw(); // Request redraw after processing command
                    }
                } else {
                    log::warn!("Received a command before state was initialized (via proxy). Ignoring: {:?}", event);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut *self.state.lock().unwrap() else {
            log::warn!("Window event received before State was initialized, ignoring.");
            return;
        };

        let window_handle = self.window.as_ref().unwrap();

        let mut needs_redraw = false;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
                needs_redraw = true;
            }
            WindowEvent::RedrawRequested => {
                if state.update() {
                    needs_redraw = true; // Still need to redraw even if update indicates change
                }
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.config.width, state.config.height),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => log::error!("{:?}", e),
                }
            }
            WindowEvent::MouseInput { state: mouse_button_state, button, .. } => {
                match (button, mouse_button_state.is_pressed()) {
                    (MouseButton::Left, true) => {
                        state.is_mouse_left_pressed = true;
                        log::info!("Mouse pos: {}, {}", state.mouse_current_pos_screen[0], state.mouse_current_pos_screen[1]);
                        state.camera.start_panning(state.mouse_current_pos_screen);
                        state.camera_needs_update = true;
                        needs_redraw = true;
                    }
                    (MouseButton::Left, false) => {
                        state.is_mouse_left_pressed = false;
                        state.camera.end_panning();
                    }
                    _ => {}
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_current_pos_screen = Vec2::new(position.x as f32, position.y as f32);
                if state.is_mouse_left_pressed {
                    state.camera.pan(state.mouse_current_pos_screen);
                    state.camera_needs_update = true;
                    needs_redraw = true;
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let y_scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => y * 10.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };

                let zoom_factor = if y_scroll_delta > 0.0 { 1.1 } else { 1.0 / 1.1 };
                let mouse_world_pos = state.camera.screen_to_world(state.mouse_current_pos_screen);
                state.camera.zoom_by(zoom_factor, mouse_world_pos);
                state.camera_needs_update = true;
                needs_redraw = true;
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        repeat,
                        ..
                    },
                ..
            } => {
                if key_state.is_pressed() && !repeat {
                    let mut changed = false;
                    let pan_speed = 10.0 / state.camera.zoom;
                    let zoom_factor = 1.1;

                    match code {
                        KeyCode::KeyW | KeyCode::ArrowUp => { state.camera.position.y += pan_speed; changed = true; },
                        KeyCode::KeyS | KeyCode::ArrowDown => { state.camera.position.y -= pan_speed; changed = true; },
                        KeyCode::KeyA | KeyCode::ArrowLeft => { state.camera.position.x -= pan_speed; changed = true; },
                        KeyCode::KeyD | KeyCode::ArrowRight => { state.camera.position.x += pan_speed; changed = true; },
                        KeyCode::KeyQ => { state.camera.zoom *= zoom_factor; changed = true; },
                        KeyCode::KeyE => { state.camera.zoom /= zoom_factor; changed = true; },
                        KeyCode::KeyR => { log::info!("FPS: {}", state.current_fps) },
                        _ => {}
                    }

                    if changed {
                        state.camera_needs_update = true;
                        needs_redraw = true;
                    }
                }
            },
            _ => {}
        }

        if needs_redraw {
            window_handle.request_redraw();
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).unwrap_throw();
        log::info!("Starting WDMView application.");
        let (sender, receiver) = flume::unbounded();
        WASM_READY_FLUME_CHANNEL.set((sender, receiver))
            .expect("Failed to initialize WASM_READY_CHANNEL. This should not happen.");
        log::info!("WASM ready channel created and stored.");
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    log::info!("WASM started: Calling run().");
    run().unwrap_throw();

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmApi {
    proxy: EventLoopProxy<UserCommand>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmApi {
    #[wasm_bindgen(js_name = setFullTopology)]
    pub fn set_full_topology(&self, topology_json: &str) -> Result<(), JsValue> {
        let parsed_topology: FullTopologyData = serde_json::from_str(topology_json)
            .map_err(|e| JsValue::from_str(&format!("JSON parsing error: {}", e)))?;

        let command = UserCommand::SetFullTopology {
            nodes: parsed_topology.nodes,
            links: parsed_topology.links,
        };

        log::info!("Received SetFullTopology command from JS.");

        if self.proxy.send_event(command).is_err() {
            return Err(JsValue::from_str("Failed to send command to event loop."));
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getWasmApi)]
pub fn get_wasm_api() -> Result<WasmApi, JsValue> {
    WASM_API_INSTANCE.get()
        .cloned()
        .ok_or_else(|| JsValue::from_str("WasmApi is not initialized. Call run_web() first."))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getWasmReadyPromise)]
pub fn get_wasm_ready_promise() -> Result<Promise, JsValue> {
    let (_, receiver) = WASM_READY_FLUME_CHANNEL.get()
        .ok_or_else(|| JsValue::from_str("WASM ready channel already taken or not initialized. Make sure getWasmApi() is called only once."))?;

    // Convert the Rust Future obtained from the flume receiver into a js_sys::Promise
    let ready_promise = future_to_promise(async move {
        receiver.recv_async().await.unwrap_throw(); // Wait for the signal
        Ok(JsValue::NULL) // Resolve with null
    });

    // 将 Rust Future 转换为 JS Promise
    Ok(ready_promise)
}

