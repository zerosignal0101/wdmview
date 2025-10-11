// src/models.rs
use bytemuck::{Pod, Zeroable};

// --- Standard 2D Vertex (for basic shapes like quads) ---
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
}

impl Vertex2D {
    // 定义一个单位四边形的四个顶点及其顺序
    pub const QUAD_VERTICES: [Self; 4] = [
        Vertex2D { position: [-0.5, -0.5] }, // 0: Bottom-left
        Vertex2D { position: [ 0.5, -0.5] }, // 1: Bottom-right
        Vertex2D { position: [ 0.5,  0.5] }, // 2: Top-right
        Vertex2D { position: [-0.5,  0.5] }, // 3: Top-left
    ];

    // 定义绘制四边形所需的索引 (两个三角形)
    pub const QUAD_INDICES: [u16; 6] = [
        0, 1, 2, // First triangle: BL, BR, TR
        0, 2, 3, // Second triangle: BL, TR, TL
    ];

    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0, // location 0 for base quad position
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}


// --- Instance Data for Circles (Nodes) ---
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CircleInstance {
    pub position: [f32; 2], // 节点中心的世界坐标
    pub radius_scale: f32,  // 节点半径 (世界单位)
    pub color: [f32; 4],    // RGBA 颜色 (线性空间)
}

impl CircleInstance {
    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance, // 步进模式为实例
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1, // location 1 for instance position
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 2, // location 2 for instance radius
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: (mem::size_of::<[f32; 2]>() + mem::size_of::<f32>())
                        as wgpu::BufferAddress,
                    shader_location: 3, // location 3 for instance color
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// --- Vertex Data for Lines (Connections) ---
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 2], // 顶点世界坐标
    pub color: [f32; 4],    // RGBA 颜色 (线性空间)
}

impl LineVertex {
    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, // location 0 for line vertex position
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1, // location 1 for line vertex color
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
