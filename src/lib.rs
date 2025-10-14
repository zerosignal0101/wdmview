use std::{collections::HashMap, sync::Arc, sync::Mutex};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey, SmolStr},
    window::Window,
};
use instant::Instant;
use glam::{Vec2, Mat4};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

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

use models::{Vertex2D, CircleInstance, LineVertex};
use camera::{Camera, CameraUniform};
use color::Color;

const BASE_NODE_RADIUS: f32 = 25.0;
const LINES_WGSL: &str = include_str!("./shaders/lines.wgsl");
const CIRCLES_WGSL: &str = include_str!("./shaders/circles.wgsl");

#[cfg(target_arch = "wasm32")]
static WASM_API_INSTANCE: OnceCell<WasmApi> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
static WASM_READY_FLUME_CHANNEL: OnceCell<(flume::Sender<()>, flume::Receiver<()>)> = OnceCell::new();

#[derive(Debug)]
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,

    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_uniform: CameraUniform,
    camera_needs_update: bool,

    line_render_pipeline: wgpu::RenderPipeline,
    circle_render_pipeline: wgpu::RenderPipeline,

    circle_instances: Vec<CircleInstance>,
    circle_instance_buffer: wgpu::Buffer,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,

    line_vertices: Vec<LineVertex>,
    line_vertex_buffer: wgpu::Buffer,

    mouse_current_pos_screen: Vec2,
    is_mouse_left_pressed: bool,

    last_frame_instant: instant::Instant,
    frame_count_in_second: u32,
    current_fps: u32,
}

impl State {
    // Now takes Arc<Window> for setup, doesn't store it.
    async fn new(window_arc: Arc<Window>) -> anyhow::Result<State> {
        let size = window_arc.inner_size();

        let gpu = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        // Surface itself is !Send on WASM due to HtmlCanvasElement
        let surface = gpu.create_surface(window_arc).unwrap();

        let adapter = gpu
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let texture_format = surface_caps.formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| {
                log::warn!("No sRGB surface format found, falling back to {:?}", surface_caps.formats[0]);
                surface_caps.formats[0]
            });

        log::info!(
            "Using {} ({:?}, Preferred Format: {:?})",
            adapter_info.name,
            adapter_info.backend,
            texture_format,
        );

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: texture_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ... (camera, pipelines, initial data setup remains the same)

        let mut camera = Camera::new(size.width, size.height);
        let camera_uniform = CameraUniform {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
        };

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("Camera Bind Group Layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("Camera Bind Group"),
        });

        // --- 着色器模块 ---
        let lines_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lines Shader"),
            source: wgpu::ShaderSource::Wgsl(LINES_WGSL.into()),
        });

        let circles_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Circles Shader"),
            source: wgpu::ShaderSource::Wgsl(CIRCLES_WGSL.into()),
        });

        // --- 渲染管线布局 ---
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // --- 线段渲染管线 ---
        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &lines_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[
                    LineVertex::layout(),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &lines_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // --- 圆形渲染管线 ---
        let circle_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Circle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &circles_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex2D::layout(),
                    CircleInstance::layout(),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &circles_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // --- 初始图形数据准备 (示例) ---
        let circle_instances = vec![
            CircleInstance {
                position: [-200.0, 0.0].into(),
                radius_scale: BASE_NODE_RADIUS,
                color: Color::from((255, 0, 0)).into_linear_rgba(),
            },
            CircleInstance {
                position: [0.0, 0.0].into(),
                radius_scale: BASE_NODE_RADIUS,
                color: Color::from((0, 255, 0)).into_linear_rgba(),
            },
            CircleInstance {
                position: [200.0, 0.0].into(),
                radius_scale: BASE_NODE_RADIUS,
                color: Color::from((0, 0, 255)).into_linear_rgba(),
            },
            CircleInstance {
                position: [0.0, 150.0].into(),
                radius_scale: BASE_NODE_RADIUS * 1.5,
                color: Color::from((255, 200, 0)).into_linear_rgba(),
            },
        ];

        let circle_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Circle Instance Buffer"),
                contents: bytemuck::cast_slice(&circle_instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let quad_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Quad Vertex Buffer"),
                contents: bytemuck::cast_slice(Vertex2D::QUAD_VERTICES.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let quad_index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Quad Index Buffer"),
                contents: bytemuck::cast_slice(Vertex2D::QUAD_INDICES.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let line_vertices = vec![
            LineVertex { position: circle_instances[0].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[1].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[1].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[2].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[0].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[3].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
        ];

        let line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        Ok( Self {
            surface, device, queue, config, is_surface_configured: false,
            camera, camera_buffer, camera_bind_group, camera_uniform, camera_needs_update: true,
            line_render_pipeline, circle_render_pipeline,
            circle_instances, circle_instance_buffer, quad_vertex_buffer, quad_index_buffer,
            line_vertices, line_vertex_buffer,
            mouse_current_pos_screen: Vec2::ZERO, is_mouse_left_pressed: false,
            last_frame_instant: Instant::now(), frame_count_in_second: 0, current_fps: 0,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            log::info!("Resize {}, {}", width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.camera.update_aspect_ratio(width, height);
            self.camera_needs_update = true;
            self.is_surface_configured = true;
            // No request_redraw here, it's App's responsibility
        }
    }

    fn update(&mut self) -> bool {
        if self.camera_needs_update {
            self.camera_uniform.view_proj = self.camera.build_view_projection_matrix().to_cols_array_2d();
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );
            self.camera_needs_update = false;
            return true;
        }
        false
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // --- FPS Calculation ---
        self.frame_count_in_second += 1;
        let now = Instant::now();
        let elapsed = (now - self.last_frame_instant).as_secs_f32();

        if elapsed >= 1.0 {
            self.current_fps = self.frame_count_in_second;
            self.frame_count_in_second = 0;
            self.last_frame_instant = now;
        }
        // --- End FPS Calculation ---

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::from((18, 18, 18)).into_linear_wgpu_color()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.line_vertices.len() as u32, 0..1);

            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.circle_instance_buffer.slice(..));
            render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..Vertex2D::QUAD_INDICES.len() as u32,
                0,
                0..self.circle_instances.len() as u32,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn process_command(&mut self, command: UserCommand) {
        match command {
            UserCommand::SetFullTopology { nodes, links } => {
                log::info!("Setting full topology with {} nodes and {} links.", nodes.len(), links.len());

                let node_id_to_idx: HashMap<u32, usize> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| (node.id, i))
                    .collect();

                self.circle_instances = nodes
                    .into_iter()
                    .map(|node| CircleInstance {
                        position: node.position.into(),
                        radius_scale: node.radius_scale,
                        color: Color::from((node.color[0], node.color[1], node.color[2])).into_linear_rgba(),
                    })
                    .collect();

                self.line_vertices.clear();
                for link in links {
                    if let (Some(&source_idx), Some(&target_idx)) = (
                        node_id_to_idx.get(&link.source_id),
                        node_id_to_idx.get(&link.target_id),
                    ) {
                        let line_color = Color::from((link.color[0], link.color[1], link.color[2]));
                        self.line_vertices.push(LineVertex {
                            position: self.circle_instances[source_idx].position,
                            color: line_color.into_linear_rgba(),
                        });
                        self.line_vertices.push(LineVertex {
                            position: self.circle_instances[target_idx].position,
                            color: line_color.into_linear_rgba(),
                        });
                    } else {
                        log::warn!("Link references non-existent node ID. Source: {}, Target: {}", link.source_id, link.target_id);
                    }
                }

                self.update_gpu_buffers();
            }
            UserCommand::AddNode(node_data) => {
                // Implement add node logic
                log::info!("Add node command received: {:?}", node_data);
            }
            UserCommand::RemoveNode(node_id) => {
                // Implement remove node logic
                log::info!("Remove node command received: {:?}", node_id);
            }
            UserCommand::StateInitialized => {
                // This command is handled in App::user_event
            }
        }
    }

    fn update_gpu_buffers(&mut self) {
        let circle_data = bytemuck::cast_slice(&self.circle_instances);
        let line_data = bytemuck::cast_slice(&self.line_vertices);

        if self.circle_instance_buffer.size() < circle_data.len() as u64 {
            self.circle_instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Circle Instance Buffer (Resized)"),
                contents: circle_data,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.circle_instance_buffer, 0, circle_data);
        }

        if self.line_vertex_buffer.size() < line_data.len() as u64 {
            self.line_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer (Resized)"),
                contents: line_data,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.line_vertex_buffer, 0, line_data);
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeData {
    pub id: u32,
    pub position: [f32; 2],
    pub radius_scale: f32,
    pub color: [u8; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinkData {
    pub source_id: u32,
    pub target_id: u32,
    pub color: [u8; 3],
}

#[derive(Deserialize, Debug)]
pub struct FullTopologyData {
    pub nodes: Vec<NodeData>,
    pub links: Vec<LinkData>,
}

#[derive(Debug)]
enum UserCommand {
    SetFullTopology {
        nodes: Vec<NodeData>,
        links: Vec<LinkData>,
    },
    AddNode(NodeData),
    RemoveNode(u32),
    StateInitialized, // Notifies App that State setup is complete
}


pub struct App {
    window: Option<Arc<Window>>, // App now owns the Arc<Window>
    state: Arc<Mutex<Option<State>>>, // Wrapped in Arc<Mutex> for interior mutability and potential Send (if State itself were Send)
    #[cfg(target_arch = "wasm32")]
    proxy: Option<EventLoopProxy<UserCommand>>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<UserCommand>) -> Self {
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
        self.window = Some(window.clone()); // App now holds the Arc<Window> instance

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

        let window_handle = self.window.as_ref().unwrap(); // Use App's window handle

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
#[derive(Clone, Debug)] // Added Debug for better logging
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
    // We take the channel from the static OnceCell. This means this function can only be called once
    // to obtain the promise. Subsequent calls would return an error.
    let (_, receiver) = WASM_READY_FLUME_CHANNEL.get()
        .ok_or_else(|| JsValue::from_str("WASM ready channel already taken or not initialized. Make sure getWasmApi() is called only once."))?;

    // Convert the Rust Future obtained from the flume receiver into a j_sys::Promise
    let ready_promise = future_to_promise(async move {
        receiver.recv_async().await.unwrap_throw(); // Wait for the signal
        Ok(JsValue::NULL) // Resolve with null
    });

    // 将 Rust Future 转换为 JS Promise
    Ok(ready_promise)
}

