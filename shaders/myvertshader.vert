#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in uint in_material;

layout(location = 0) out vec3 out_pos;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out flat uint out_material;


layout(push_constant) uniform PushConstants {
    mat4 VP;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer ModelMatrices {
    mat4 M[];
} models;


void main() {
    mat4 M = models.M[gl_InstanceIndex];

    vec4 world_pos = M * vec4(in_pos, 1.0);
    gl_Position = pc.VP * world_pos;

    out_pos = world_pos.xyz;
    out_uv = in_uv;
    out_normal = transpose(inverse(mat3(M))) * in_normal;
    out_material = in_material;
}
