#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec4 v_color;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec4 color;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    v_color = uniforms.color;
}
