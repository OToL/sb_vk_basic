#version 450

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;
layout(location=2) in vec2 in_tex_coords;

layout(binding=0) uniform  UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 projection;
}uni_mvp;

layout(location=0) out vec3 out_color;
layout(location=1) out vec2 out_tex_coords;

void main ()
{
    gl_Position = uni_mvp.projection * uni_mvp.view * uni_mvp.model * vec4(in_position, 1.0);
    out_color = in_color;
    out_tex_coords = in_tex_coords;
}
