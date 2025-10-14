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
                let world_visible_height = (2.0 / self.zoom);

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
}
