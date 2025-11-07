use std::{collections::HashMap, sync::Arc, sync::Mutex};
use winit::{
    event::*,
    window::Window,
};
use instant::Instant;
use glam::Vec2;
use bevy_color::{ColorToComponents, LinearRgba, Oklcha, Srgba};
use wgpu::util::DeviceExt;


use crate::models::{Vertex2D, CircleInstance, LineVertex};
use crate::camera::{Camera, CameraUniform};
use crate::scene::connection::ConnectionData;
use crate::scene::defrag_event::{reconstruct_state_at_time, AnyEvent};
use crate::scene::service::ServiceData; // 引入 ServiceData
use crate::scene::element::ElementData;
use crate::scene::text_label::TextLabel; // 引入 ElementData


pub const BASE_NODE_RADIUS: f32 = 20.0;
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
    pub glyphon_buffers: Vec<glyphon::Buffer>,

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

    // --- 新增时间轴和拓扑数据管理字段 ---
    pub all_elements: Vec<ElementData>, // 存储所有节点数据
    pub all_connections: Vec<ConnectionData>,
    pub all_events: Vec<AnyEvent>, // 存储所有事件变化数据
    // 用于快速查找节点 ID 对应的 circle_instances 索引
    pub node_id_to_idx: HashMap<String, usize>,
    pub current_time_selection: f32, // 当前时间轴选中的时刻

    pub highlight_service_id_list: Option<Vec<i32>>, // 当前选中的碎片整理过程，围绕这一 id，需要高亮
    pub highlight_line_render_pipeline: wgpu::RenderPipeline, // 新增高亮线路渲染管线
    pub highlight_line_vertices: Vec<LineVertex>,             // 新增高亮线路顶点数据
    pub highlight_line_vertex_buffer: wgpu::Buffer,           // 新增高亮线路顶点缓冲区
    pub highlight_node_color: [f32; 4], // 高亮节点的颜色
    pub world_text_labels: Vec<TextLabel>,

    pub topology_needs_update: bool, // 标记拓扑（主要是服务线路）是否需要因时间变化而更新

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
        let color_mode = if needs_shader_srgb_output_conversion {
            glyphon::ColorMode::Web
        } else {
            glyphon::ColorMode::Accurate
        };
        let mut glyphon_atlas = glyphon::TextAtlas::with_color_mode(&device, &queue, &glyphon_cache, texture_format, color_mode);
        let glyphon_renderer = glyphon::TextRenderer::new(&mut glyphon_atlas, &device, wgpu::MultisampleState::default(), None);

        // Create text buffers
        let buffer_num = 4000 as usize;
        let mut glyphon_buffers = Vec::with_capacity(buffer_num);
        for _i in 0..buffer_num {
            let text_buffer = glyphon::Buffer::new(&mut glyphon_font_system, glyphon::Metrics::relative(10.0, 16.0));
            glyphon_buffers.push(text_buffer);
        }
        
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
                color: LinearRgba::from(Srgba::rgb_u8(255, 0, 0)).to_f32_array(),
            },
            CircleInstance {
                position: [0.0, 0.0].into(),
                radius_scale: BASE_NODE_RADIUS,
                color: LinearRgba::from(Srgba::rgb_u8(0, 255, 0)).to_f32_array(),
            },
            CircleInstance {
                position: [200.0, 0.0].into(),
                radius_scale: BASE_NODE_RADIUS,
                color: LinearRgba::from(Srgba::rgb_u8(0, 0, 255)).to_f32_array(),
            },
            CircleInstance {
                position: [0.0, 150.0].into(),
                radius_scale: BASE_NODE_RADIUS * 1.5,
                color: LinearRgba::from(Srgba::rgb_u8(255, 200, 0)).to_f32_array(),
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
            LineVertex { position: circle_instances[0].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
            LineVertex { position: circle_instances[1].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
            LineVertex { position: circle_instances[1].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
            LineVertex { position: circle_instances[2].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
            LineVertex { position: circle_instances[0].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
            LineVertex { position: circle_instances[3].position.into(), color: LinearRgba::from(Srgba::rgb_u8(200, 200, 200)).to_f32_array() },
        ];

        let line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        // --- 高亮线段着色器模块 ---
        let highlight_lines_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Highlight Lines Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/highlight_lines.wgsl").into()),
        });

        // --- 高亮线段渲染管线 ---
        let highlight_line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Highlight Line Render Pipeline"),
            layout: Some(&render_pipeline_layout), // 共用布局
            vertex: wgpu::VertexState {
                module: &highlight_lines_shader_module,
                entry_point: Some("vs_main"), // 可以是与 lines.wgsl 相同的 vs_main
                buffers: &[
                    LineVertex::layout(), // 同样使用 LineVertex 布局
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &highlight_lines_shader_module,
                entry_point: Some("fs_main"), // 可以是与 lines.wgsl 相同的 fs_main
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 关键：使用 TriangleList
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // 双面渲染，因为四边形可能被裁剪
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

        let highlight_line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Highlight Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&[] as &[LineVertex]), // 初始为空
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        Ok( Self {
            surface, device, queue, config, is_surface_configured: false,
            glyphon_font_system, glyphon_swash_cache, glyphon_viewport,
            glyphon_atlas, glyphon_renderer, glyphon_buffers,
            camera, camera_buffer, camera_bind_group, camera_uniform, camera_needs_update: true,
            line_render_pipeline, circle_render_pipeline,
            circle_instances, circle_instance_buffer, quad_vertex_buffer, quad_index_buffer,
            line_vertices, line_vertex_buffer,
            mouse_current_pos_screen: Vec2::ZERO, is_mouse_left_pressed: false,
            last_frame_instant: Instant::now(), frame_count_in_second: 0, current_fps: 0,
            // --- 新增字段初始化 ---
            all_elements: Vec::new(),
            all_connections: Vec::new(),
            all_events: Vec::new(),
            node_id_to_idx: HashMap::new(),
            current_time_selection: 0.0, // 默认初始时间为 0
            highlight_service_id_list: None,
            highlight_line_render_pipeline,
            highlight_line_vertices: Vec::new(),
            highlight_line_vertex_buffer,
            highlight_node_color: LinearRgba::from(Srgba::rgb_u8(0xd2, 0xa1, 0x06)).to_f32_array(), // 黄色 40
            world_text_labels: Vec::new(),
            topology_needs_update: false,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            log::info!("Resize {}, {}", width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            // Update glyphon buffer size
            for glyphon_buffer in self.glyphon_buffers.iter_mut() {
                glyphon_buffer.set_size(
                    &mut self.glyphon_font_system,
                    Some(width as f32),
                    Some(height as f32),
                );
                glyphon_buffer.shape_until_scroll(&mut self.glyphon_font_system, false);
            }

            self.camera.update_aspect_ratio(width, height);
            self.camera_needs_update = true;
            self.is_surface_configured = true;
            // No request_redraw here, it's App's responsibility
        }
    }

    pub fn update(&mut self) -> bool {
        let mut needs_redraw = false;

        if self.camera_needs_update {
            self.camera_uniform.view_proj = self.camera.build_view_projection_matrix().to_cols_array_2d();
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );
            self.camera_needs_update = false;
            needs_redraw = true;
        }
        
        // 如果拓扑（主要是服务线路）需要更新
        if self.topology_needs_update {
            log::debug!("Updating topology due to time change or initial load. Time: {}", self.current_time_selection);
            self.generate_all_lines_for_current_time();
            self.update_gpu_buffers(); // Upload new line vertices to GPU
            self.topology_needs_update = false;
            needs_redraw = true; // Request redraw to show updated lines
        }

        needs_redraw
    }

    pub fn update_gpu_buffers(&mut self) {
        let circle_data = bytemuck::cast_slice(&self.circle_instances);
        let line_data = bytemuck::cast_slice(&self.line_vertices);

        // (Re)create circle instance buffer if size changes, otherwise write
        if self.circle_instance_buffer.size() < circle_data.len() as u64 {
            self.circle_instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Circle Instance Buffer (Resized)"),
                contents: circle_data,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.circle_instance_buffer, 0, circle_data);
        }

        // (Re)create line vertex buffer if size changes, otherwise write
        if self.line_vertex_buffer.size() < line_data.len() as u64 {
            self.line_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer (Resized)"),
                contents: line_data,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.line_vertex_buffer, 0, line_data);
        }

        let highlight_line_data = bytemuck::cast_slice(&self.highlight_line_vertices);
        if self.highlight_line_vertex_buffer.size() < highlight_line_data.len() as u64 {
            self.highlight_line_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Highlight Line Vertex Buffer (Resized)"),
                contents: highlight_line_data,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            self.queue.write_buffer(&self.highlight_line_vertex_buffer, 0, highlight_line_data);
        }
    }

    // Helper to generate a thick line (quad) from two points
    fn add_thick_line_segment(
        &mut self,
        start_pos: Vec2,
        end_pos: Vec2,
        color: [f32; 4],
        thickness: f32, // 世界单位厚度
    ) {
        let dir = end_pos - start_pos;
        let length = dir.length();

        if length < f32::EPSILON {
            return; // Avoid division by zero for zero-length lines
        }

        let normalized_dir = dir.normalize();
        let perpendicular_dir = Vec2::new(-normalized_dir.y, normalized_dir.x); // 旋转90度

        let half_thickness_offset = perpendicular_dir * (thickness / 2.0); // 注意：厚度需要反比例于缩放，以在屏幕上保持一致的像素宽度

        let p1_minus_offset = start_pos - half_thickness_offset;
        let p1_plus_offset = start_pos + half_thickness_offset;
        let p2_plus_offset = end_pos + half_thickness_offset;
        let p2_minus_offset = end_pos - half_thickness_offset;

        // 添加构成两个三角形的六个顶点
        self.highlight_line_vertices.push(LineVertex { position: p1_minus_offset.into(), color });
        self.highlight_line_vertices.push(LineVertex { position: p1_plus_offset.into(), color });
        self.highlight_line_vertices.push(LineVertex { position: p2_plus_offset.into(), color }); // Triangle 1: (p1-, p1+, p2+)

        self.highlight_line_vertices.push(LineVertex { position: p1_minus_offset.into(), color });
        self.highlight_line_vertices.push(LineVertex { position: p2_plus_offset.into(), color });
        self.highlight_line_vertices.push(LineVertex { position: p2_minus_offset.into(), color }); // Triangle 2: (p1-, p2+, p2-)
    }

    /// 根据当前时间轴选择，重新生成所有链接和服务的线条。
    fn generate_all_lines_for_current_time(&mut self) {
        self.line_vertices.clear();
        self.highlight_line_vertices.clear(); // 清除高亮线条数据

        let radius_inside = BASE_NODE_RADIUS;
        const LINK_BOUNDARY_ROTATE_ANGLE: f32 = std::f32::consts::PI / 16.0;
        const HIGHLIGHT_LINE_THICKNESS: f32 = 0.5; // 世界单位厚度
        const NORMAL_LINE_COLOR: [f32; 4] = [0.784, 0.784, 0.784, 1.0]; // 灰色，从 Srgba::rgb_u8(200, 200, 200).to_f32_array()

        // 追踪所有被高亮服务触及的节点ID
        let mut nodes_in_highlighted_services: std::collections::HashSet<String> = std::collections::HashSet::new();
        if let Some(ref highlight_ids) = self.highlight_service_id_list {
            let reconstructed_service_dict = reconstruct_state_at_time(&self.all_events, self.current_time_selection);
            for service_id in highlight_ids {
                if let Some(service) = reconstructed_service_dict.get(service_id) {
                    // Collect all nodes in path for highlighting
                    for node_id in &service.path {
                        nodes_in_highlighted_services.insert(node_id.clone());
                    }
                }
            }
        }

        // --- 1. 更新节点颜色 ---
        // 首先恢复所有节点为默认颜色
        for instance in self.circle_instances.iter_mut() {
            instance.color = LinearRgba::from(Srgba::rgb_u8(0x00, 0x5d, 0x5d)).to_f32_array();
        }
        // 然后根据高亮列表重新着色
        for (node_id, &instance_idx) in &self.node_id_to_idx {
            if nodes_in_highlighted_services.contains(node_id) {
                self.circle_instances[instance_idx].color = self.highlight_node_color;
            }
        }


        // --- 2. 渲染固定的链路边界 (普通细线) ---
        for link in &self.all_connections {
            if let (Some(&source_idx), Some(&target_idx)) = (
                self.node_id_to_idx.get(&link.from_node),
                self.node_id_to_idx.get(&link.to_node),
            ) {
                let link_boundary_color = LinearRgba::from(Srgba::rgb_u8(180, 180, 180));
                let source_position_center = Vec2::from_array(self.circle_instances[source_idx].position);
                let destination_position_center = Vec2::from_array(self.circle_instances[target_idx].position);
                let dir_vec = destination_position_center - source_position_center;
                let length = dir_vec.length();

                if length < f32::EPSILON {
                    continue;
                }

                let normalized_dir = dir_vec.normalize();
                let radius_dir_outward = normalized_dir * radius_inside;

                let rotate_vector = Vec2::from_angle(LINK_BOUNDARY_ROTATE_ANGLE);
                let reverse_rotate_vector = Vec2::from_angle(-LINK_BOUNDARY_ROTATE_ANGLE);

                self.line_vertices.push(LineVertex {
                    position: (source_position_center + radius_dir_outward.rotate(rotate_vector)).into(),
                    color: link_boundary_color.to_f32_array(),
                });
                self.line_vertices.push(LineVertex {
                    position: (destination_position_center - radius_dir_outward.rotate(reverse_rotate_vector)).into(),
                    color: link_boundary_color.to_f32_array(),
                });

                self.line_vertices.push(LineVertex {
                    position: (source_position_center + radius_dir_outward.rotate(reverse_rotate_vector)).into(),
                    color: link_boundary_color.to_f32_array(),
                });
                self.line_vertices.push(LineVertex {
                    position: (destination_position_center - radius_dir_outward.rotate(rotate_vector)).into(),
                    color: link_boundary_color.to_f32_array(),
                });
            } else {
                log::warn!("Link references non-existent node ID. Source: {}, Target: {}", link.from_node, link.to_node);
            }
        }

        // --- 3. 渲染当前时间活跃的服务线条 ---
        const MAX_WAVELENGTHS: u32 = 80;
        const SERVICE_MAX_SPREAD_ANGLE: f32 = LINK_BOUNDARY_ROTATE_ANGLE * 0.95;

        let reconstructed_service_dict = reconstruct_state_at_time(&self.all_events, self.current_time_selection);

        for (service_id, service) in reconstructed_service_dict.iter() {
            let departure_time = service.departure_time;
            // 检查服务是否在当前时间活跃
            if self.current_time_selection >= service.arrival_time && self.current_time_selection < departure_time {
                let wavelength = service.wavelength;
                let effective_wavelength = (wavelength as f32).min((MAX_WAVELENGTHS - 1) as f32);

                let hue_color = (effective_wavelength + 0.5) / (MAX_WAVELENGTHS as f32) * 180.0 + 30.0;

                let is_highlighted = match &self.highlight_service_id_list {
                    Some(highlight_service_id_list) => highlight_service_id_list.iter().any(|&srv_id| srv_id == *service_id),
                    None => false,
                };

                let service_color_oklcha = if is_highlighted {
                    // 高亮服务的颜色可以更鲜明，例如保持高饱和度，但亮度适中，或者采用完全不同的颜色
                    Oklcha::lch(0.75, 0.2, hue_color) // 更亮的颜色
                } else {
                    if self.highlight_service_id_list.iter().len() == 0{
                        Oklcha::lch(0.6, 0.11, hue_color)
                    }
                    else {
                        Oklcha::lch(0.4, 0.11, hue_color)
                    }
                };
                let service_color_f32 = LinearRgba::from(service_color_oklcha).to_f32_array();
                // 如果不是高亮服务，亮度调整回默认的0.6。
                // `service_color_f32` will be determined by `is_highlighted`.

                let normalized_wavelength_factor = (effective_wavelength - ((MAX_WAVELENGTHS as f32 - 1.0) / 2.0)) / ((MAX_WAVELENGTHS as f32 - 1.0) / 2.0);
                let wavelength_rotate_angle = normalized_wavelength_factor * SERVICE_MAX_SPREAD_ANGLE;

                for i in 0..(service.path.len() - 1) {
                    let source_node_id = &service.path[i];
                    let target_node_id = &service.path[i + 1];

                    if let (Some(&source_idx), Some(&target_idx)) = (
                        self.node_id_to_idx.get(source_node_id),
                        self.node_id_to_idx.get(target_node_id),
                    ) {
                        let source_pos_center = Vec2::from_array(self.circle_instances[source_idx].position);
                        let target_pos_center = Vec2::from_array(self.circle_instances[target_idx].position);

                        let dir_vec = target_pos_center - source_pos_center;
                        let length = dir_vec.length();

                        if length < f32::EPSILON {
                            continue;
                        }

                        let normalized_dir = dir_vec.normalize();
                        let radius_vec_along_link = normalized_dir * radius_inside;

                        let upward_sacle: f32 = if normalized_dir.y >= 0.0 { 1.0 } else { -1.0 };
                        let service_start_pos = source_pos_center + radius_vec_along_link.rotate(Vec2::from_angle(wavelength_rotate_angle * upward_sacle));
                        let service_end_pos = target_pos_center - radius_vec_along_link.rotate(Vec2::from_angle( - wavelength_rotate_angle * upward_sacle));

                        if is_highlighted {
                            self.add_thick_line_segment(service_start_pos, service_end_pos, service_color_f32, HIGHLIGHT_LINE_THICKNESS);
                            self.world_text_labels.push(TextLabel { content: format!("{}", i), radius_scale: BASE_NODE_RADIUS, position: source_pos_center.into() });
                            if i == service.path.len() - 2 {
                                self.world_text_labels.push(TextLabel { content: format!("{}", i + 1), radius_scale: BASE_NODE_RADIUS, position: target_pos_center.into() });
                            }
                        } else {
                            self.line_vertices.push(LineVertex { position: service_start_pos.into(), color: service_color_f32 });
                            self.line_vertices.push(LineVertex { position: service_end_pos.into(), color: service_color_f32 });
                        }
                    } else {
                        log::warn!(
                            "Service {} path references non-existent node ID. Segment: {} -> {}",
                            service_id, source_node_id, target_node_id
                        );
                    }
                }

                // Processing the segments inside the circle (if any)
                for i in 0..(service.path.len() - 2) {
                    let source_node_id = &service.path[i];
                    let middle_node_id = &service.path[i + 1];
                    let target_node_id = &service.path[i + 2];

                    if let (Some(&source_idx), Some(&middle_idx), Some(&target_idx)) = (
                        self.node_id_to_idx.get(source_node_id),
                        self.node_id_to_idx.get(middle_node_id),
                        self.node_id_to_idx.get(target_node_id),
                    ) {
                        let source_pos_center = Vec2::from_array(self.circle_instances[source_idx].position);
                        let middle_pos_center = Vec2::from_array(self.circle_instances[middle_idx].position);
                        let target_pos_center = Vec2::from_array(self.circle_instances[target_idx].position);

                        let source_middle_dir_vec = target_pos_center - middle_pos_center;
                        let middle_target_dir_vec = middle_pos_center - source_pos_center;

                        let normalized_source_middle_dir = source_middle_dir_vec.normalize();
                        let normalized_middle_target_dir = middle_target_dir_vec.normalize();

                        let radius_source_middle_vec_along_link = normalized_source_middle_dir * radius_inside;
                        let radius_middle_target_vec_along_link = normalized_middle_target_dir * radius_inside;

                        let source_middle_upward_sacle: f32 = if normalized_source_middle_dir.y >= 0.0 { 1.0 } else { -1.0 };
                        let middle_target_upward_sacle: f32 = if normalized_middle_target_dir.y >= 0.0 { 1.0 } else { -1.0 };

                        let middle_start_pos = middle_pos_center + radius_source_middle_vec_along_link.rotate(Vec2::from_angle(wavelength_rotate_angle * source_middle_upward_sacle));
                        let middle_end_pos = middle_pos_center - radius_middle_target_vec_along_link.rotate(Vec2::from_angle( - wavelength_rotate_angle * middle_target_upward_sacle));

                        if is_highlighted {
                            self.add_thick_line_segment(middle_start_pos, middle_end_pos, service_color_f32, HIGHLIGHT_LINE_THICKNESS);
                        } else {
                            self.line_vertices.push(LineVertex { position: middle_start_pos.into(), color: service_color_f32 });
                            self.line_vertices.push(LineVertex { position: middle_end_pos.into(), color: service_color_f32 });
                        }
                    } else {
                        log::warn!(
                            "Service {} path references non-existent node ID. Segment: {} -> {} -> {}",
                            service_id, source_node_id, middle_node_id, target_node_id
                        );
                    }
                }
            }
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        // Update glyphon viewport
        let width = self.config.width;
        let height = self.config.height;

        if width == 0 || height == 0 {
            log::warn!("Attempting to render with zero width or height, skipping.");
            return Ok(());
        }

        self.glyphon_viewport.update(&self.queue, glyphon::Resolution { width, height });

        // --- Prepare Glyphon Text Areas ---
        let mut text_areas = Vec::new();

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

        // 获取相机在世界坐标中可见的区域，用于粗粒度裁剪
        let (world_visible_min, world_visible_max) = self.camera.get_world_clip_bounds();

        // Node Labels (e.g., radius)
        for (i, (instance, glyphon_buffer)) in self.world_text_labels.iter().zip(self.glyphon_buffers.iter_mut()).enumerate() {
            // 1. 粗粒度世界坐标裁剪
            if instance.position[0] < world_visible_min.x - instance.radius_scale * 2.0 || // 加上半径的裕量
               instance.position[0] > world_visible_max.x + instance.radius_scale * 2.0 ||
               instance.position[1] < world_visible_min.y - instance.radius_scale * 2.0 ||
               instance.position[1] > world_visible_max.y + instance.radius_scale * 2.0 {
                continue; // 节点超出世界可见范围，不渲染文本
            }

            let screen_pos = self.camera.world_to_screen(instance.position.into());
            let screen_radius = self.camera.world_radius_to_screen_pixels(instance.radius_scale);

            // 3. 级别细节 (LOD) 裁剪：如果节点太小，不显示标签
            const MIN_DISPLAY_SCREEN_RADIUS: f32 = 60.0;
            if screen_radius < MIN_DISPLAY_SCREEN_RADIUS {
                continue;
            }

            // --- 动态字体大小和定位 ---
            let target_base_font_size_world = 8.0; // 世界坐标系下，文本的“理想”高度单位
            let actual_font_size_screen = target_base_font_size_world * self.camera.zoom * (self.config.height as f32 / 2.0);
            let clamped_font_size = actual_font_size_screen.clamp(10.0, 40.0); // 限制字体大小在合理范围

            let label_text = &instance.content; // 文本内容

            // 只有当文本内容、字体大小或布局参数变化时才更新 TextBuffer
            // 否则，Glyphon会使用其内部缓存
            // 此处无法直接检测文本内容变化，所以如果每次都格式化字符串，则假定每次都可能变
            // 真正的 dirty flag 应该包含文本内容的 hash 或引用
            let metrics = glyphon::Metrics::new(clamped_font_size, clamped_font_size * 1.2); // 行高稍大一点
            
            glyphon_buffer.set_metrics(&mut self.glyphon_font_system, metrics);
            glyphon_buffer.set_size(
                &mut self.glyphon_font_system,
                Some(screen_radius), // 给一个足够宽的矩形来防止不必要的换行，或者计算实际可用宽度
                None, // 不需要固定高度，让 Glyphon 自动计算
            );
            glyphon_buffer.set_text(
                &mut self.glyphon_font_system,
                label_text,
                &glyphon::Attrs::new().family(glyphon::Family::SansSerif),
                glyphon::Shaping::Advanced,
            );
            glyphon_buffer.shape_until_scroll(&mut self.glyphon_font_system, false);

            // 获取文本的实际宽度以便准确居中
            let mut text_width = 0.0;
            let mut text_height = 0.0;
            if let Some(run) = glyphon_buffer.layout_runs().next() {
                text_width = run.line_w;
                text_height = run.line_height * glyphon_buffer.layout_runs().count() as f32; // Sum of all line heights
            }

            // 根据屏幕半径和实际文本大小调整位置
            let text_left = screen_pos.x - text_width / 2.0; // 文本中心与节点中心对齐
            let text_top = screen_pos.y - text_height / 2.0; // 文本放在节点上方，留 5 像素间距

            // 将文本区域添加到待渲染列表
            text_areas.push(glyphon::TextArea {
                buffer: glyphon_buffer,
                left: text_left,
                top: text_top,
                scale: 1.0, // scale 1.0 是指 buffer 内部的字体大小已经是最终屏幕尺寸
                bounds: glyphon::TextBounds::default(), // 可以在这里设置裁剪矩形
                default_color: glyphon::Color::rgb(230, 230, 230),
                custom_glyphs: &[]
            });
        }

        // Prepare glyphon text for rendering (uploads glyph textures)
        self.glyphon_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.glyphon_font_system,
            &mut self.glyphon_atlas,
            &self.glyphon_viewport,
            text_areas, // Pass the vector of TextAreas
            &mut self.glyphon_swash_cache,
        ).unwrap();

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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // 1. 绘制圆形（节点）
            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.circle_instance_buffer.slice(..));
            render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..Vertex2D::QUAD_INDICES.len() as u32,
                0,
                0..self.circle_instances.len() as u32,
            );

            // 2. 绘制普通线段 (链路边界和服务)
            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.line_vertices.len() as u32, 0..1);

            // 3. 绘制高亮线段 (覆盖在普通线段之上)
            if self.highlight_line_vertices.len() != 0 {
                render_pass.set_pipeline(&self.highlight_line_render_pipeline);
                render_pass.set_vertex_buffer(0, self.highlight_line_vertex_buffer.slice(..));
                render_pass.draw(0..self.highlight_line_vertices.len() as u32, 0..1);
            }
            
            // --- Draw Glyphon Text ---
            self.glyphon_renderer.render(&self.glyphon_atlas, &self.glyphon_viewport, &mut render_pass).unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.glyphon_atlas.trim();

        Ok(())
    }

        /// 根据当前拓扑（`circle_instances`）调整相机位置和缩放，使其全部可见。
    pub fn fit_view_to_topology(&mut self) {
        if self.circle_instances.is_empty() {
            // 如果没有节点，则将相机重置到默认视图
            self.camera.position = glam::Vec2::ZERO;
            self.camera.zoom = 1.0;
            self.camera_needs_update = true;
            return;
        }

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut max_node_radius = 0.0f32;

        // 遍历所有节点实例以确定边界
        for instance in &self.circle_instances {
            min_x = min_x.min(instance.position[0]);
            max_x = max_x.max(instance.position[0]);
            // 注意：拓扑数据中的y坐标在加载时反转了 (-element.metadata.location.y)
            // 所以这里直接使用 instance.position[1] 来计算世界坐标的 min/max Y
            min_y = min_y.min(instance.position[1]);
            max_y = max_y.max(instance.position[1]);
            max_node_radius = max_node_radius.max(instance.radius_scale);
        }

        // 为了确保节点完全可见，扩大边界框，考虑到最大的节点半径
        // 增加额外的边距，防止节点被裁剪
        const PADDING_MULTIPLIER: f32 = 1.2; // 增加20%的额外空间
        let padded_min_x = min_x - max_node_radius * PADDING_MULTIPLIER;
        let padded_max_x = max_x + max_node_radius * PADDING_MULTIPLIER;
        let padded_min_y = min_y - max_node_radius * PADDING_MULTIPLIER;
        let padded_max_y = max_y + max_node_radius * PADDING_MULTIPLIER;

        let bounding_box_width = padded_max_x - padded_min_x;
        let bounding_box_height = padded_max_y - padded_min_y;

        // 如果边界框尺寸过小（例如只有一个节点），设定一个最小可见尺寸以避免无限缩放
        const MIN_VISIBLE_WORLD_DIM: f32 = 200.0; // 最小世界单位尺寸
        let target_world_width = bounding_box_width.max(MIN_VISIBLE_WORLD_DIM);
        let target_world_height = bounding_box_height.max(MIN_VISIBLE_WORLD_DIM);
        
        // 计算所需的缩放级别，以适应宽度和高度
        let mut zoom_x = 1.0;
        if self.camera.aspect_ratio > f32::EPSILON && target_world_width > f32::EPSILON {
            zoom_x = (2.0 * self.camera.aspect_ratio) / target_world_width;
        }

        let mut zoom_y = 1.0;
        if target_world_height > f32::EPSILON {
            zoom_y = 2.0 / target_world_height;
        }

        // 为了确保所有内容都可见，我们选择两者中较小的缩放值（即更“缩小”的视图）
        let new_zoom = zoom_x.min(zoom_y).clamp(0.001, 1000.0); // 限制缩放范围

        // 设置新的相机中心位置
        self.camera.position = glam::Vec2::new(
            (padded_min_x + padded_max_x) / 2.0,
            (padded_min_y + padded_max_y) / 2.0,
        );
        self.camera.zoom = new_zoom;
        self.camera_needs_update = true; // 标记相机需要更新
        log::info!("View fitted to topology. New camera position: {:?}, zoom: {}", self.camera.position, self.camera.zoom);
    }
}
