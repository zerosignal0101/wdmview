// src/lib.rs
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey, SmolStr}, // Added SmolStr for `Character` event
    window::Window,
};
use glam::{Vec2, Mat4}; // 引入 glam 类型
use wgpu::util::DeviceExt; // 用于 buffer_init_descriptor

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod models;
mod camera; // 引入 camera 模块
mod color; // Color 模块已经存在

use models::{Vertex2D, CircleInstance, LineVertex};
use camera::{Camera, CameraUniform};
use color::Color;

// 定义节点的基础半径（世界单位）
const BASE_NODE_RADIUS: f32 = 25.0;

// Shaders 作为字符串字面量嵌入
const LINES_WGSL: &str = include_str!("./shaders/lines.wgsl");
const CIRCLES_WGSL: &str = include_str!("./shaders/circles.wgsl");

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,

    // 相机相关
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_uniform: CameraUniform, // 存储当前帧的 uniform 数据
    camera_needs_update: bool, // 标记相机是否需要更新 uniform buffer

    // 渲染管线
    line_render_pipeline: wgpu::RenderPipeline,
    circle_render_pipeline: wgpu::RenderPipeline,

    // 节点 (圆形) 数据
    circle_instances: Vec<CircleInstance>,
    circle_instance_buffer: wgpu::Buffer, // 实例数据缓冲区
    quad_vertex_buffer: wgpu::Buffer,    // 基础四边形顶点缓冲区
    quad_index_buffer: wgpu::Buffer,     // 基础四边形索引缓冲区

    // 连接 (线段) 数据
    line_vertices: Vec<LineVertex>,
    line_vertex_buffer: wgpu::Buffer,

    // 鼠标输入状态
    mouse_current_pos_screen: Vec2, // 鼠标当前屏幕坐标
    is_mouse_left_pressed: bool,      // 鼠标左键是否按下
    // is_mouse_right_pressed: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        let gpu = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = gpu.create_surface(window.clone()).unwrap();

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
                // 为了更好的兼容性，特别是 WebGL -> WebGPU
                // 确保支持了 `shader-f16` (可能并非所有设备都支持) 和 `texture-compression-bc`,
                // 具体特性根据需求添加，这里保持空以最大兼容。
                required_features: wgpu::Features::empty(), // wgpu::Features::NON_FILL_POLYGON_MODE for Wireframe
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, // Trace path
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // 优先选择 sRGB 格式，如果可用 (通常是 `Rgba8UnormSrgb`)
        let texture_format = surface_caps.formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            // 如果找不到 sRGB 格式，那么就用默认的第一个，但这可能会出现颜色不匹配问题
            // 考虑在此处直接 panic 或 fallback 到一个已知行为的安全非sRGB格式，并手动在shader中转换
            .unwrap_or_else(|| {
                // Warning: If no sRGB format is found, colors might still look off 
                // unless you manually convert to sRGB in the fragment shader.
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
        surface.configure(&device, &config); // 首次配置 surface

        // --- 相机设置 ---
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
                    visibility: wgpu::ShaderStages::VERTEX, // 相机矩阵只在顶点着色器中使用
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
                &camera_bind_group_layout, // 绑定组 0 用于相机
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
                    LineVertex::layout(), // 仅需要 LineVertex 的布局
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &lines_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // 启用 Alpha 混合
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList, // 绘制线条列表 (每两个顶点构成一条线)
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // 线条通常不需要背面剔除
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1, // 禁用多重采样
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
                    Vertex2D::layout(),      // 基础四边形顶点数据
                    CircleInstance::layout(), // 每实例数据
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &circles_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // 启用 Alpha 混合以实现圆形平滑边缘
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // 剔除四边形的背面
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1, // 禁用多重采样
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
                radius_scale: BASE_NODE_RADIUS * 1.5, // 大一些的节点
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

        // 线路连接示例
        let line_vertices = vec![
            // 线 1: 节点 0 <-> 节点 1
            LineVertex { position: circle_instances[0].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[1].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            // 线 2: 节点 1 <-> 节点 2
            LineVertex { position: circle_instances[1].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            LineVertex { position: circle_instances[2].position.into(), color: Color::from((200, 200, 200)).into_linear_rgba() },
            // 线 3: 节点 0 <-> 节点 3
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
            window,
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,

            camera,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            camera_needs_update: true,

            line_render_pipeline,
            circle_render_pipeline,

            circle_instances,
            circle_instance_buffer,
            quad_vertex_buffer,
            quad_index_buffer,

            line_vertices,
            line_vertex_buffer,

            mouse_current_pos_screen: Vec2::ZERO,
            is_mouse_left_pressed: false,
        })
    }

    fn window(&self) -> &Window {
        &self.window
    }

    /// 窗口大小改变时调用
    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            // 通知相机更新宽高比
            self.camera.update_aspect_ratio(width, height);
            self.camera_needs_update = true;
            self.window.request_redraw(); // 请求重绘
        }
    }

    /// 更新渲染数据（目前只更新相机）
    fn update(&mut self) -> bool {
        // 如果相机需要更新，则重新计算 view_proj 矩阵并写入 buffer
        if self.camera_needs_update {
            self.camera_uniform.view_proj = self.camera.build_view_projection_matrix().to_cols_array_2d();
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );
            self.camera_needs_update = false;
            return true; // 状态已更新，需要重绘
        }
        false
    }

    /// 执行渲染操作
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // 如果 surface 尚未配置，则不能渲染
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::from((18, 18, 18)).into_linear_wgpu_color()), // 背景色：深灰
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 绑定相机 Uniform
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // --- 绘制线段 ---
            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.line_vertices.len() as u32, 0..1);

            // --- 绘制圆形 (实例渲染) ---
            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..)); // 绑定基础四边形顶点
            render_pass.set_vertex_buffer(1, self.circle_instance_buffer.slice(..)); // 绑定实例数据
            render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16); // 绑定索引
            render_pass.draw_indexed(
                0..Vertex2D::QUAD_INDICES.len() as u32, // 绘制基础四边形的索引范围
                0,
                0..self.circle_instances.len() as u32, // 为每个实例绘制
            );
        } // `render_pass` 结束，因为它需要 `encoder` 的可变借用

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<Option<State>>>, // proxy now sends Option<State>
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<Option<State>>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<Option<State>> for App { // UserEvent is now Option<State>
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes()
            .with_title("WDMView Graph Topology"); // Set window title

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

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut state = pollster::block_on(State::new(window)).unwrap(); // Add mut here
            let current_size = state.window().inner_size();
            state.resize(current_size.width, current_size.height); // <-- 强制初始化尺寸
            state.window().request_redraw();
            self.state = Some(state);
        }

        #[cfg(target_arch = "wasm32")]
        {
            // For wasm, `resumed` is called once the app starts.
            // We need to pass the window Arc to the async task which creates state
            // and then sends it back to the main thread via the proxy.
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    let state_result = State::new(window.clone()).await; // Clone window for async context
                    match state_result {
                        Ok(state) => {
                            if proxy.send_event(Some(state)).is_err() {
                                log::error!("Failed to send new State via proxy (channel closed).");
                            }
                        },
                        Err(e) => {
                            log::error!("Failed to create State: {:?}", e);
                            if proxy.send_event(None).is_err() { // Send None to indicate failure
                                log::error!("Failed to send error state via proxy.");
                            }
                        }
                    }
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: Option<State>) {
        // This is where State created in WASM async task is received
        // Or if error, None is received
        if let Some(mut state_instance) = event.take() { // take() the Option<State> out
            let current_size = state_instance.window().inner_size();
            state_instance.resize(current_size.width, current_size.height); // <-- 强制初始化尺寸
            log::info!("WASM State initialized!");
            state_instance.window().request_redraw(); // Request redraw after init
            self.state = Some(state_instance); // Assign after resize and redraw
        } else {
            log::error!("Failed to initialize State in user_event (WASM).");
            // Optionally, handle error, e.g., display a message to the user
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(s) => s,
            None => {
                log::warn!("Window event received before State was initialized, ignoring.");
                return;
            },
        };

        let mut needs_redraw = false;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
                needs_redraw = true; // Resize implicitly requires redraw
            }
            WindowEvent::RedrawRequested => {
                if state.update() { // If update causes a state change like camera movement affecting uniform buffer
                    // A true from update already means a redraw is needed, and `update` internally
                    // already writes to the camera buffer.
                }
                match state.render() {
                    Ok(_) => {}
                    // 当 surface 丢失，可能是因为窗口设置改变或设备被拔出。
                    // 这种情况通常需要重新配置 surface。
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.config.width, state.config.height),
                    // 当系统内存不足时
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    // 其他所有错误（比如 `Timeout` 和 `Outdated`）的发生都表明 surface 需要重新配置但不是紧急情况
                    Err(e) => log::error!("{:?}", e),
                }
            }
            WindowEvent::MouseInput { state: mouse_button_state, button, .. } => {
                match (button, mouse_button_state.is_pressed()) {
                    (MouseButton::Left, true) => {
                        state.is_mouse_left_pressed = true;
                        // 开始拖拽平移，记录鼠标当前世界坐标作为参考点
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
                    // 如果正在拖拽，进行平移操作
                    state.camera.pan(state.mouse_current_pos_screen);
                    state.camera_needs_update = true;
                    needs_redraw = true;
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let y_scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => y * 10.0, // 典型滚动，转换为更显著的量
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32, // 精确像素滚动
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
                if key_state.is_pressed() && !repeat { // Only trigger on initial press
                    // Example: Basic keyboard pan/zoom for testing
                    let mut changed = false;
                    let pan_speed = 10.0 / state.camera.zoom; // Move faster when zoomed out
                    let zoom_factor = 1.1;

                    match code {
                        KeyCode::KeyW | KeyCode::ArrowUp => { state.camera.position.y += pan_speed; changed = true; },
                        KeyCode::KeyS | KeyCode::ArrowDown => { state.camera.position.y -= pan_speed; changed = true; },
                        KeyCode::KeyA | KeyCode::ArrowLeft => { state.camera.position.x -= pan_speed; changed = true; },
                        KeyCode::KeyD | KeyCode::ArrowRight => { state.camera.position.x += pan_speed; changed = true; },
                        KeyCode::KeyQ => { state.camera.zoom *= zoom_factor; changed = true; },
                        KeyCode::KeyE => { state.camera.zoom /= zoom_factor; changed = true; },
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
            state.window.request_redraw();
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
        console_log::init_with_level(log::Level::Info).unwrap_throw();
        log::info!("Success init wasm.");
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
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
