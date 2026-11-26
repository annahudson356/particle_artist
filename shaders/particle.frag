#version 450

//-------------------------------------------------------------------------------------//
// PART K: PARTICLE FRAGMENT SHADER
//-------------------------------------------------------------------------------------//
// A point rasterizes as a square. lightpoint.frag shows how to make it a circle. Your
// fragment shader should contain the final required feature, which is to have particles
// interact with scene lights. At minimum, a particle's brightness will fall off with its
// distance from each light. Particles have no normals by default, so you cannot create
// the entire BRDF, at least without further features. You are reusing the
// graphics_descriptor_sets so check myfragshader.frag for how to define the Lights
// buffer (same goes for the Light struct).

layout(location = 0) in flat vec3 in_pos;

layout(location = 0) out vec4 fragColor;

struct Light {
    vec4 position;
    vec3 color;
    float intensity;
};

layout(std430, set = 0, binding = 1) readonly buffer Lights {
    vec4 camera_pos;
    uint light_count;
    float A;
    mat4 sun_VP;
    vec4 sun_direction;
    vec4 sun_color;
    Light light[];
} lights;
void main()
{
    vec2 from_center = gl_PointCoord - vec2(0.5);
    if (dot(from_center, from_center) > 0.25) discard;

    vec3 kd = vec3(1.0);
    vec3 color = lights.A * kd;

    for (uint i = 0; i < lights.light_count; ++i) {
        Light light = lights.light[i];

        float d = length(light.position.xyz - in_pos);
        vec3 radiance = light.color.rgb * light.intensity / (1.0 + d * d);

        color += radiance * kd;
    }
    fragColor = vec4(color, 1.0);
}

