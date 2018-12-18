#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 in_frag_color;
layout(location = 1) in vec2 in_frag_text;

layout(location = 0) out vec4 out_frag_color;

layout(binding=1) uniform sampler2D text_sampler;

void main()
{
    //out_frag_color = vec4(in_frag_color, 1.0);
    //out_frag_color = vec4(in_frag_text, 0.0, 1.0);
    out_frag_color = texture(text_sampler, in_frag_text);
}
