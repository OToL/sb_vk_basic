#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding=1) uniform sampler2D tex_sampler;

layout(location=0) in vec3 frag_color;
layout(location=1) in vec2 tex_coords;

layout(location=0) out vec4 out_color;

void main()
{
    // make it brighter
    out_color = vec4(1.5 * texture(tex_sampler, tex_coords).rgb, 1.0);
}
