// src/camera.rs
// 画布放大、移动
use glam::{Mat4, Vec2, Vec3, Vec4}; // 引入 glam 库的向量和矩阵类型
use glam::Vec4Swizzles;
use bytemuck::{Pod, Zeroable};

// 将发送到 GPU 的相机 Uniform 数据结构
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4], // 视图投影矩阵
    pub needs_srgb_output_conversion: u32, // 0 for false, 1 for true
    pub _padding: [u32; 3], // 填充到 16 字节边界，使 CameraUniform 总大小为 80 字节
}

#[derive(Debug)]
pub struct Camera {
    pub position: Vec2, // 相机在世界坐标中的中心点
    pub zoom: f32,      // 缩放级别。1.0 为默认，>1.0 放大，<1.0 缩小。
    pub aspect_ratio: f32, // 视口宽高比 (width / height)
    pub viewport_size: Vec2, // 视口的像素尺寸

    // 鼠标交互状态
    is_panning: bool,
    last_mouse_pos_screen: Option<Vec2>, // 上次鼠标位置 (屏幕坐标) 用于拖拽平移
}

impl Camera {
    pub fn new(viewport_width: u32, viewport_height: u32) -> Self {
        let aspect_ratio = viewport_width as f32 / viewport_height as f32;
        Self {
            position: Vec2::ZERO, // 默认相机位于世界坐标原点
            zoom: 1.0,           // 默认缩放
            aspect_ratio: if aspect_ratio.is_finite() && aspect_ratio > 0.0 { aspect_ratio } else { 1.0 },
            viewport_size: Vec2::new(viewport_width as f32, viewport_height as f32),
            is_panning: false,
            last_mouse_pos_screen: None,
        }
    }

    /// 更新视口的宽高比和像素尺寸，在窗口大小改变时调用
    pub fn update_aspect_ratio(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.aspect_ratio = width as f32 / height as f32;
            self.viewport_size = Vec2::new(width as f32, height as f32);
        }
    }

    /// 将屏幕坐标 (像素，左上角为原点) 转换为世界坐标
    pub fn screen_to_world(&self, screen_coords: Vec2) -> Vec2 {
        if self.viewport_size.x == 0.0 || self.viewport_size.y == 0.0 {
            return Vec2::ZERO;
        }

        // 将屏幕坐标归一化到 NDC (Normalized Device Coordinates) 范围 [-1, 1]
        // 屏幕 Y 轴向下为正，NDC 和世界 Y 轴向上为正，因此需要反转 Y
        let ndc_x = (screen_coords.x / self.viewport_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_coords.y / self.viewport_size.y) * 2.0;

        // 使用逆视图投影矩阵将 NDC 转换回世界坐标
        let view_proj_inv = self.build_view_projection_matrix().inverse();
        let world_coords_vec4 = view_proj_inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);

        // 归一化 w 分量（尽管对于正交投影 w 应该为 1.0）
        world_coords_vec4.xy() / world_coords_vec4.w
    }

    /// 将世界坐标点转换为屏幕像素坐标
    /// 返回值 Vec2 的 x, y 是像素值，原点在左上角
    pub fn world_to_screen(&self, world_coords: glam::Vec2) -> glam::Vec2 {
        if self.viewport_size.x == 0.0 || self.viewport_size.y == 0.0 {
            return Vec2::ZERO;
        }
        // 1. 将世界坐标点转换为齐次坐标 (z=0, w=1 for 2D)
        let world_coords_vec4 = glam::Vec4::new(world_coords.x, world_coords.y, 0.0, 1.0);
        // 2. 将世界坐标通过视图投影矩阵转换为裁剪空间坐标
        let clip_pos_vec4 = self.build_view_projection_matrix() * world_coords_vec4;
        // 3. 执行透视除法（对于正交投影，w 通常为 1，但保留通用性）
        let ndc_pos = clip_pos_vec4.xy() / clip_pos_vec4.w;
        // 4. 将 NDC (-1 to 1) 转换为屏幕坐标 (0 to width/height)
        let screen_x = (ndc_pos.x * 0.5 + 0.5) * self.viewport_size.x as f32;
        // 注意：NDC 的 Y 轴方向通常是向上为正，而屏幕坐标 Y 轴是向下为正
        // 所以需要 1.0 - (ndc_pos.y * 0.5 + 0.5) 来进行 Y 轴翻转
        let screen_y = (1.0 - (ndc_pos.y * 0.5 + 0.5)) * self.viewport_size.y as f32;
        glam::Vec2::new(screen_x, screen_y)
    }

    /// 将世界空间半径转换为屏幕像素半径
    /// 假设正交投影，并且 x/y 轴缩放一致（由相机宽高比处理）
    pub fn world_radius_to_screen_pixels(&self, world_radius: f32) -> f32 {
        // 世界空间中，可视区域的高度是 2.0 / camera.zoom
        world_radius * (self.viewport_size.y as f32 * self.zoom / 2.0)
    }

    /// 开始平移操作
    pub fn start_panning(&mut self, screen_pos: Vec2) {
        self.is_panning = true;
        self.last_mouse_pos_screen = Some(screen_pos);
    }

    /// 执行平移操作
    pub fn pan(&mut self, current_screen_pos: Vec2) {
        if self.is_panning {
            if let Some(last_pos) = self.last_mouse_pos_screen {
                let screen_delta = current_screen_pos - last_pos;

                // 计算每个像素在世界坐标中的实际距离
                let world_visible_width = (2.0 / self.zoom) * self.aspect_ratio;
                let world_visible_height = 2.0 / self.zoom;

                let world_units_per_pixel_x = world_visible_width / self.viewport_size.x;
                let world_units_per_pixel_y = world_visible_height / self.viewport_size.y;

                let world_delta_x = screen_delta.x * world_units_per_pixel_x;
                let world_delta_y = screen_delta.y * world_units_per_pixel_y;

                // 更新相机位置。鼠标向右移动 (screen_delta.x > 0)，相机（视图）向左移动 (position.x 减小)
                // 鼠标向下移动 (screen_delta.y > 0)，相机（视图）向上移动 (position.y 增大，因为世界 Y 轴向上)
                self.position.x -= world_delta_x;
                self.position.y += world_delta_y;
            }
            self.last_mouse_pos_screen = Some(current_screen_pos);
        }
    }

    /// 结束平移操作
    pub fn end_panning(&mut self) {
        self.is_panning = false;
        self.last_mouse_pos_screen = None;
    }

    /// 根据一个因子进行缩放，并保持 `world_focus` 点在世界坐标中不动
    pub fn zoom_by(&mut self, factor: f32, world_focus: Vec2) {
        let old_zoom = self.zoom;
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(0.001, 1000.0); // 限制缩放范围，防止过大或过小

        // 调整相机位置以保持焦点不变
        let offset = self.position - world_focus; // 获取焦点到相机中心的向量
        // 根据缩放比例反向调整这个向量，然后加回到焦点上得到新的相机位置
        self.position = world_focus + offset / (self.zoom / old_zoom);
    }

    /// 构建视图投影矩阵
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        // 正交投影矩阵: 定义世界空间中可见的区域。
        // `zoom` 参数直接影响这个可见区域的大小。Zoom 越大，可见世界区域越小，即视觉效果上 "放大"。
        let half_world_width = self.aspect_ratio / self.zoom;
        let half_world_height = 1.0 / self.zoom;

        // Mat4::orthographic_rh(left, right, bottom, top, near, far)
        // 这个投影将指定的方形世界体积映射到 NDC (x,y,z 均为 [-1,1])。
        let proj_matrix = Mat4::orthographic_rh(
            -half_world_width,  // left
            half_world_width,   // right
            -half_world_height, // bottom
            half_world_height,  // top
            -100.0,             // near (对于 2D 而言是任意的，但提供了绘制顺序的深度)
            100.0,              // far
        );

        // 视图矩阵: 转换世界坐标系到相机坐标系。
        // 对于仅平移的 2D 相机，这是一个平移矩阵。
        // 它会移动世界数据，使得相机的 `position` 成为相机视图的原点。
        let view_matrix = Mat4::from_translation(Vec3::new(-self.position.x, -self.position.y, 0.0));

        // 组合视图投影矩阵
        proj_matrix * view_matrix
    }

    pub fn get_world_clip_bounds(&self) -> (Vec2, Vec2) {
        let half_world_width = self.aspect_ratio / self.zoom;
        let half_world_height = 1.0 / self.zoom;

        let center = self.position;

        let min_x = center.x - half_world_width;
        let max_x = center.x + half_world_width;
        let min_y = center.y - half_world_height;
        let max_y = center.y + half_world_height;

        (Vec2::new(min_x, min_y), Vec2::new(max_x, max_y))
    }
}
