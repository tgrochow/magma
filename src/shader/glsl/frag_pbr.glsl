#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 10.0, 10.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = 0.5 * vec3(v_color[0], v_color[1], v_color[2]);
    vec3 regular_color = 1.0 * vec3(v_color[0], v_color[1], v_color[2]);

    f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}
