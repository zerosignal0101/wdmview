// src/shaders/highlight_lines.wgsl
// 通用相机 Uniform 结构
struct CameraUniform {
    view_proj: mat4x4<f32>,
    needs_srgb_output_conversion: u32,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// 转换线性颜色到 sRGB 颜色
fn linear_to_srgb(c: f32) -> f32 {
    if c < 0.0031308 {
        return c * 12.92;
    } else {
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
}

// 顶点着色器输入结构 (对应 LineVertex)
struct LineVertexInput {
    @location(0) position: vec2<f32>, // 顶点世界坐标
    @location(1) color: vec4<f32>,    // 顶点颜色
};

// 片元着色器输入结构
struct LineFragmentInput {
    @builtin(position) clip_position: vec4<f32>, // 裁剪空间位置
    @location(0) color: vec4<f32>,               // 传递给片元着色器的颜色
};

@vertex
fn vs_main(
    model: LineVertexInput,
) -> LineFragmentInput {
    var out: LineFragmentInput;
    // 将世界坐标转换为裁剪空间坐标
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color; // 直接传递顶点颜色
    return out;
}

@fragment
fn fs_main(in: LineFragmentInput) -> @location(0) vec4<f32> {
    var final_color = in.color;
    if camera.needs_srgb_output_conversion == 1u {
        final_color.r = linear_to_srgb(final_color.r);
        final_color.g = linear_to_srgb(final_color.g);
        final_color.b = linear_to_srgb(final_color.b);
    }
    return final_color;
}
