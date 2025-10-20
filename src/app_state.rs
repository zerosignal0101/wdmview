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


const BASE_NODE_RADIUS: f32 = 25.0;
const LINES_WGSL: &str = include_str!("./shaders/lines.wgsl");
const CIRCLES_WGSL: &str = include_str!("./shaders/circles.wgsl");

#[derive(Debug, Clone)]
pub struct ChannelLink {
    pub start_node_idx: usize, // 起始节点的索引 (对应 circle_instances 的索引)
    pub end_node_idx: usize,   // 结束节点的索引
    pub num_bands: u32,        // 波段数量
    pub band_spacing: f32,     // 每个波段之间的世界单位距离
}


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

    pub connections: Vec<ChannelLink>, // 存储通道链接数据
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

        // --- 初始化通道连接数据 ---
        let connections = vec![
            ChannelLink { start_node_idx: 0, end_node_idx: 1, num_bands: 3, band_spacing: 12.0 },
            ChannelLink { start_node_idx: 1, end_node_idx: 2, num_bands: 5, band_spacing: 8.0 },
            ChannelLink { start_node_idx: 0, end_node_idx: 3, num_bands: 2, band_spacing: 15.0 },
            ChannelLink { start_node_idx: 2, end_node_idx: 3, num_bands: 4, band_spacing: 10.0 },
        ];

        // --- 根据节点和连接生成所有波段的线段顶点 ---
        let line_vertices = Self::generate_band_vertices(&circle_instances, &connections);

        let line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                // 使用 COPY_DST 允许后续帧更新线段数据
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
            connections, line_vertices, line_vertex_buffer,
            mouse_current_pos_screen: Vec2::ZERO, is_mouse_left_pressed: false,
            last_frame_instant: Instant::now(), frame_count_in_second: 0, current_fps: 0,
        })
    }

    /// 根据当前节点和连接数据生成所有波段的线段顶点。
    fn generate_band_vertices(
        nodes: &[CircleInstance],
        links: &[ChannelLink],
    ) -> Vec<LineVertex> {
        let mut all_line_vertices = Vec::new();

        // 提取节点的世界坐标，方便查找
        let node_positions: Vec<Vec2> = nodes.iter().map(|n| n.position.into()).collect();

        for link in links {
            let start_pos_center = node_positions[link.start_node_idx];
            let end_pos_center = node_positions[link.end_node_idx];

            let dir_vec = end_pos_center - start_pos_center;
            let length = dir_vec.length();

            // 避免除以零或非常短的线段
            if length < f32::EPSILON {
                continue;
            }

            let normalized_dir = dir_vec.normalize();
            // 计算垂直于连接方向的向量。对于 2D 向量 (x, y)，(-y, x) 是一个垂直向量。
            let perpendicular_dir = Vec2::new(-normalized_dir.y, normalized_dir.x);

            for i in 0..link.num_bands {
                // 计算每个波段的偏移量。
                // 这里的逻辑使得 num_bands 为奇数时，中间波段（i = (num_bands - 1) / 2）恰好在节点中心连线上。
                // 如果 num_bands 为偶数，则没有波段在中心线上，它们会对称分布。
                let offset_factor = (i as f32) - ((link.num_bands - 1) as f32 / 2.0);
                let offset = perpendicular_dir * offset_factor * link.band_spacing;

                let band_start_pos = start_pos_center + offset;
                let band_end_pos = end_pos_center + offset;

                // 根据载波编号（这里简化为波段索引 i）生成颜色
                // 你可以根据实际需求设计更复杂的颜色生成逻辑
                let band_color_f32 = Oklcha::lch(0.6, 0.092, 10.0 * (i as f32)).to_f32_array();

                all_line_vertices.push(LineVertex { position: band_start_pos.into(), color: band_color_f32 });
                all_line_vertices.push(LineVertex { position: band_end_pos.into(), color: band_color_f32 });
            }
        }
        all_line_vertices
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

        // Update glyphon viewport
        let width = self.config.width;
        let height = self.config.height;
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
        for (i, (instance, glyphon_buffer)) in self.circle_instances.iter().zip(self.glyphon_buffers.iter_mut()).enumerate() {
            // 1. 粗粒度世界坐标裁剪
            // 简单判断节点中心是否在世界可见范围内
            // 更精确的做法是判断节点的边界框与世界可见区域是否相交
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
            let target_base_font_size_world = 3.0; // 世界坐标系下，文本的“理想”高度单位
            let actual_font_size_screen = target_base_font_size_world * self.camera.zoom * (self.config.height as f32 / 2.0);
            let clamped_font_size = actual_font_size_screen.clamp(10.0, 40.0); // 限制字体大小在合理范围

            let label_text = format!("ID: {} \n Radius: {:.1}", i, screen_radius); // 文本内容

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
                &label_text,
                &glyphon::Attrs::new().family(glyphon::Family::SansSerif),
                glyphon::Shaping::Advanced,
            );
            glyphon_buffer.shape_until_scroll(&mut self.glyphon_font_system, false);

            // 获取文本的实际宽度以便准确居中
            let mut text_width = 0.0;
            let mut text_height = 0.0;
            if let Some(run) = glyphon_buffer.layout_runs().next() {
                text_width = run.line_w;
                text_height = run.line_height * glyphon_buffer.layout_runs().count() as f32;
            }

            // 根据屏幕半径和实际文本大小调整位置
            let text_left = screen_pos.x - text_width / 2.0; // 文本中心与节点中心对齐
            let text_top = screen_pos.y - text_height; // 文本放在节点上方，留 5 像素间距

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
