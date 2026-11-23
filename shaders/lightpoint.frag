#version 450

layout(location = 0) in flat vec3 in_color;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec2 from_center = gl_PointCoord - vec2(0.5);
    if (dot(from_center, from_center) > 0.25) discard;

    fragColor = vec4(in_color, 1.0);
}
