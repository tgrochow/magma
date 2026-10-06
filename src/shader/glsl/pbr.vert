#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 texture_coords;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_texture_coords;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat3 normal_matrix = transpose(inverse(mat3(uniforms.world)));
    v_normal = normalize(normal_matrix * normal);
    world_pos = (uniforms.world * vec4(position, 1.0)).xyz;
    v_texture_coords = texture_coords;
    gl_Position = uniforms.proj * uniforms.view  * uniforms.world * vec4(position, 1.0);
}
