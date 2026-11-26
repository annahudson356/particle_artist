#version 450
//-------------------------------------------------------------------------------------//
// PART J: PARTICLE VERTEX SHADER
//-------------------------------------------------------------------------------------//
// Your inputs should match the vertex attributes you established with your particle
// pipeline in PART E, locations included. You must transform your particles from
// world-space to clip-space like any other vertex. You MUST define gl_PointSize if you
// want your shader to run, since we are using points. Divide point_scale by w to achieve
// perspective-correct sizing (see how lightpoint.vert does it).


layout(location = 0) out flat vec3 out_pos; 

layout(push_constant) uniform Push {
    mat4 vp;
    float point_scale;
    float time;
} pc;

layout(location = 0) in vec3 in_position;

void main()
{
    gl_Position = pc.vp * vec4(in_position, 1.0);
    gl_PointSize = pc.point_scale / gl_Position.w;
    out_pos = in_position;
}