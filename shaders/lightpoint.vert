#version 450

layout(location = 0) out flat vec3 frag_color;

layout(push_constant) uniform PushConstants {
    mat4 VP;
    float point_scale;
} pc;

struct PointLight {
    vec4 position;  //xyz = world position
    vec4 color;     //rgb = color, a = intensity
};

layout(std430, set = 0, binding = 1) readonly buffer Lights {
    vec4 camera_pos;
    uint light_count;
    float ambient;
    mat4 sun_VP;
    vec4 sun_direction;
    vec4 sun_color;
    PointLight light[];
} lights;

void main() {
    PointLight light = lights.light[gl_VertexIndex];

    gl_Position = pc.VP * vec4(light.position.xyz, 1.0);

    gl_PointSize = pc.point_scale / gl_Position.w;

    frag_color = light.color.rgb;
}
