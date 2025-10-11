// 通用相机 Uniform 结构
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// 基础四边形顶点输入 (对应 Vertex2D)
struct QuadVertexInput {
    @location(0) quad_position: vec2<f32>, // 基础四边形的局部坐标 (-0.5 到 0.5)
};

// 实例数据输入 (对应 CircleInstance)
struct CircleInstanceInput {
    @location(1) instance_world_position: vec2<f32>, // 圆心世界坐标
    @location(2) instance_radius_scale: f32,         // 半径 (世界单位)
    @location(3) instance_color: vec4<f32>,          // 颜色
};

// 片元着色器输入结构
struct CircleFragmentInput {
    @builtin(position) clip_position: vec4<f32>, // 裁剪空间位置
    @location(0) color: vec4<f32>,               // 传递给片元着色器的颜色
    @location(1) uv: vec2<f32>,                 // 四边形内的 UV 坐标 (-1.0 到 1.0)
};

@vertex
fn vs_main(
    quad: QuadVertexInput,
    instance: CircleInstanceInput,
) -> CircleFragmentInput {
    var out: CircleFragmentInput;

    // 将基础四边形的局部坐标按实例半径缩放
    // quad_position 从 -0.5 到 0.5，乘以 radius_scale 后，表示四边形的实际半宽/高
    // 例如，如果 radius_scale 是 25.0，则四边形从 -12.5 到 12.5 (总宽 25.0)
    let scaled_quad_local_offset = quad.quad_position * instance.instance_radius_scale;

    // 将缩放后的局部偏移量加到圆心世界坐标，得到最终的世界位置
    let world_position = instance.instance_world_position + scaled_quad_local_offset;

    // 将世界坐标转换为裁剪空间坐标
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 0.0, 1.0);
    out.color = instance.instance_color;

    // 计算并传递 UV 坐标给片元着色器，范围从 -1.0 到 1.0 (方便距离计算)
    out.uv = quad.quad_position * 2.0;
    return out;
}

@fragment
fn fs_main(in: CircleFragmentInput) -> @location(0) vec4<f32> {
    // 计算 UV 坐标点到圆心的距离（单位圆内）
    let dist_from_center_squared = dot(in.uv, in.uv);
    let dist = sqrt(dist_from_center_squared);

    // 使用 smoothstep 函数进行抗锯齿处理，使圆形边缘平滑
    // 当 dist 接近 1.0 (圆边界) 时，alpha 从 1.0 平滑过渡到 0.0
    // 参数 0.98 和 1.0 定义了平滑过渡的范围
    let alpha = smoothstep(1.0, 0.98, dist);

    // 如果 alpha 极小，则丢弃该片元，优化性能（不绘制完全透明的像素）
    if alpha < 0.01 {
        discard;
    }

    // 返回经过 alpha 混合的颜色
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
