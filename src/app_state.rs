use std::{collections::HashMap, sync::Arc, sync::Mutex};
use winit::{
    event::*,
    window::Window,
};
use instant::Instant;
use glam::Vec2;
use wgpu::util::DeviceExt;


use crate::models::{Vertex2D, CircleInstance, LineVertex};
use crate::camera::{Camera, CameraUniform};
use crate::color::Color;


const BASE_NODE_RADIUS: f32 = 25.0;
const LINES_WGSL: &str = include_str!("./shaders/lines.wgsl");
const CIRCLES_WGSL: &str = include_str!("./shaders/circles.wgsl");

pub struct State {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub is_surface_configured: bool,

    // Glyphon related fields
    pub glyphon_font_system: glyphon::FontSystem,
    pub glyphon_viewport: glyphon::Viewport,
    pub glyphon_swash_cache: glyphon::SwashCache,
    pub glyphon_atlas: glyphon::TextAtlas,
    pub glyphon_renderer: glyphon::TextRenderer,
    pub glyphon_buffer: glyphon::Buffer,

    pub camera: Camera,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_uniform: CameraUniform,
    pub camera_needs_update: bool,

    pub line_render_pipeline: wgpu::RenderPipeline,
    pub circle_render_pipeline: wgpu::RenderPipeline,

    pub circle_instances: Vec<CircleInstance>,
    pub circle_instance_buffer: wgpu::Buffer,
    pub quad_vertex_buffer: wgpu::Buffer,
    pub quad_index_buffer: wgpu::Buffer,

    pub line_vertices: Vec<LineVertex>,
    pub line_vertex_buffer: wgpu::Buffer,

    pub mouse_current_pos_screen: Vec2,
    pub is_mouse_left_pressed: bool,

    pub last_frame_instant: instant::Instant,
    pub frame_count_in_second: u32,
    pub current_fps: u32,
}

impl State {
    // Now takes Arc<Window> for setup, doesn't store it.
    pub async fn new(window_arc: Arc<Window>) -> anyhow::Result<State> {
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

        // 确定是否需要着色器进行 sRGB 输出转换
        let needs_shader_srgb_output_conversion = !texture_format.is_srgb();

        log::info!(
            "Using {} ({:?}, Target Format: {:?}), Needs Shader sRGB Output Conversion: {}",
            adapter_info.name,
            adapter_info.backend,
            texture_format,
            needs_shader_srgb_output_conversion
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

        // --- Glyphon Initialization ---
        // --- Glyphon Initialization ---
        let mut glyphon_font_system = glyphon::FontSystem::new_with_fonts([
            glyphon::fontdb::Source::Binary(Arc::new(include_bytes!(
                "../assets/fonts/Iced-Icons.ttf"
            ))),
            glyphon::fontdb::Source::Binary(Arc::new(include_bytes!(
                "../assets/fonts/Roboto-Regular.ttf"
            ))),
            glyphon::fontdb::Source::Binary(Arc::new(include_bytes!(
                "../assets/fonts/bootstrap-icons.ttf"
            ))),
        ]);
        let glyphon_swash_cache = glyphon::SwashCache::new();
        let glyphon_cache = glyphon::Cache::new(&device);
        let glyphon_viewport = glyphon::Viewport::new(&device, &glyphon_cache);
        let mut glyphon_atlas = glyphon::TextAtlas::new(&device, &queue, &glyphon_cache, texture_format);
        let glyphon_renderer = glyphon::TextRenderer::new(&mut glyphon_atlas, &device, wgpu::MultisampleState::default(), None);

        // Create a text buffer
        let mut glyphon_buffer = glyphon::Buffer::new(
            &mut glyphon_font_system,
            glyphon::Metrics::new(30.0, 42.0), // Font size and line height
        );

        glyphon_buffer.set_size(
            &mut glyphon_font_system,
            Some(size.width as f32),
            Some(size.height as f32),
        );
        glyphon_buffer.set_text(
            &mut glyphon_font_system, 
            "Hello world! 👋\nThis is rendered with 🦅 glyphon 🦁\nThe text below should be partially clipped.\na b c d e f g h i j k l m n o p q r s t u v w x y z", 
            &glyphon::Attrs::new().family(glyphon::Family::SansSerif), 
            glyphon::Shaping::Advanced
        );
        glyphon_buffer.shape_until_scroll(&mut glyphon_font_system, false);
        
        #[allow(unused_mut)]
        let mut camera = Camera::new(size.width, size.height);
        let camera_uniform = CameraUniform {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
            needs_srgb_output_conversion: needs_shader_srgb_output_conversion as u32,
            _padding: [0; 3],
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
            glyphon_font_system, glyphon_swash_cache, glyphon_viewport,
            glyphon_atlas, glyphon_renderer, glyphon_buffer,
            camera, camera_buffer, camera_bind_group, camera_uniform, camera_needs_update: true,
            line_render_pipeline, circle_render_pipeline,
            circle_instances, circle_instance_buffer, quad_vertex_buffer, quad_index_buffer,
            line_vertices, line_vertex_buffer,
            mouse_current_pos_screen: Vec2::ZERO, is_mouse_left_pressed: false,
            last_frame_instant: Instant::now(), frame_count_in_second: 0, current_fps: 0,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            log::info!("Resize {}, {}", width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            // Update glyphon buffer size
            self.glyphon_buffer.set_size(
                &mut self.glyphon_font_system,
                Some(width as f32),
                Some(height as f32),
            );
            self.glyphon_buffer.shape_until_scroll(&mut self.glyphon_font_system, false);

            self.camera.update_aspect_ratio(width, height);
            self.camera_needs_update = true;
            self.is_surface_configured = true;
            // No request_redraw here, it's App's responsibility
        }
    }

    pub fn update(&mut self) -> bool {
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

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

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

        // Update glyphon viewport
        let width = self.config.width;
        let height = self.config.height;
        self.glyphon_viewport.update(&self.queue, glyphon::Resolution { width, height });

        // Update glyphon text buffer
        self.glyphon_buffer.set_text(
            &mut self.glyphon_font_system,
            &format!("FPS: {}", self.current_fps),
            &glyphon::Attrs::new().family(glyphon::Family::SansSerif),
            glyphon::Shaping::Advanced,
        );
        self.glyphon_buffer.shape_until_scroll(&mut self.glyphon_font_system, false);

        // Prepare glyphon text for rendering (uploads glyph textures)
        self.glyphon_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.glyphon_font_system,
            &mut self.glyphon_atlas,
            &self.glyphon_viewport,
            [glyphon::TextArea {
                buffer: &self.glyphon_buffer,
                left: 0.0,
                top: 5.0,
                scale: 1.0,
                bounds: glyphon::TextBounds::default(),
                default_color: glyphon::Color::rgb(255, 255, 255),
                custom_glyphs: &[]
            }],
            &mut self.glyphon_swash_cache,
        ).unwrap(); // Handle error gracefully in real app

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

            // --- Draw Glyphon Text ---
            self.glyphon_renderer.render(&self.glyphon_atlas, &self.glyphon_viewport, &mut render_pass).unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.glyphon_atlas.trim();

        Ok(())
    }
}
