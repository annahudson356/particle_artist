#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in flat uint in_material;


layout(location = 0) out vec4 fragColor;

struct Material{
    vec4 diffuse;
    vec3 specular;
    float shininess;
    uint has_diffuse_texture;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct Light {
    vec4 position;
    vec3 color;
    float intensity;
};

layout(std430, set = 0, binding = 1) readonly buffer Lights {
    vec4 camera_pos;
    uint light_count;
    float A;
    // Added for Point light shadow
    uint which_shadow;          
    float shadow_far_plane; 

    mat4 sun_VP;
    vec4 sun_direction;
    vec4 sun_color;
    Light light[];
} lights;

layout(set = 0, binding = 2) uniform sampler2DShadow shadow_map;

// Added for point light shadow
layout(set = 0, binding = 3) uniform samplerCubeShadow point_shadow; 

layout(std430, set = 1, binding = 0) readonly buffer Materials {
    Material material[];
}materials;

layout(set = 1, binding = 1) uniform sampler2D diffuse_texture;
const float POINT_SHADOW_NEAR = 0.05; 


float point_visibility(vec3 p, vec3 light_pos)
{
    vec3 v = p - light_pos; 
    float d = max(max(abs(v.x), abs(v.y)), abs(v.z));
    if (d >= lights.shadow_far_plane) return 1.0;
    d = max(d - 0.01, POINT_SHADOW_NEAR);
    float ref = lights.shadow_far_plane * (d - POINT_SHADOW_NEAR) / (d * (lights.shadow_far_plane - POINT_SHADOW_NEAR));
    return texture(point_shadow, vec4(v, ref));
}
void main()
{
    Material mat = materials.material[in_material];
    vec3 kd = mat.diffuse.rgb;
    vec3 ks = mat.specular.rgb;
    float ns = max(mat.shininess, 1.0);

    vec3 texel = texture(diffuse_texture, in_uv).rgb;
    if (mat.has_diffuse_texture != 0u) kd = texel;

    vec3 n = normalize(in_normal);
    vec3 v = normalize(lights.camera_pos.xyz - in_pos);

    vec3 color = lights.A * kd;

    for (uint i = 0; i < lights.light_count; ++i) {
        Light light = lights.light[i];

        vec3 to_light = light.position.xyz - in_pos;
        float d = length(to_light);
        vec3 l = normalize(to_light); //to_light / d

        float ndotl = max(dot(n, l), 0.0); //frag alignment
        vec3 diffuse = kd * ndotl;

        vec3 h = normalize(l + v);
        float ndoth = max(dot(n, h), 0.0); //facet alignment
        float spec_str = pow(ndoth, ns);
        vec3 specular = ks * spec_str;

        if (ndotl <= 0.0) specular = vec3(0.0);

        vec3 radiance = light.color.rgb * light.intensity / (1 + pow(d, 2));

        // Updated for point light shadow, commented code was in original shader
        // color += radiance * (diffuse + specular);
        if (i == lights.which_shadow) {
            radiance *= point_visibility(in_pos + n * 0.03, light.position.xyz);
        }

        color += radiance * (diffuse + specular);
    }

    vec3 sun_l = normalize(-lights.sun_direction.xyz);
    float sun_ndotl = max(dot(n, sun_l), 0.0);
    vec3 sun_h = normalize(sun_l + v);
    vec3 sun_specular = ks * pow(max(dot(n, sun_h), 0.0), ns);
    if (sun_ndotl <= 0.0) sun_specular = vec3(0.0);

    vec4 sun_clip = lights.sun_VP * vec4(in_pos, 1.0);
    vec3 sun_ndc = sun_clip.xyz / sun_clip.w;
    float sun_visibility = texture(shadow_map, vec3(sun_ndc.xy * 0.5 + 0.5, sun_ndc.z));
    if (sun_ndc.z > 1.0) sun_visibility = 1.0;

    color += lights.sun_color.rgb * lights.sun_color.a * (kd * sun_ndotl + sun_specular) * sun_visibility;

    fragColor = vec4(color, 1.0);
}
