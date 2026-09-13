#version 450

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform CameraData {
    vec3 pos;
} camera;

layout(push_constant) uniform PushConstantData {
    vec4 color;
    float mettalic;
    float roughness;
} material;

const vec3 lightPos = vec3(2.0, 4.0, 0.0);
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera.pos - world_pos);
    vec3 L = normalize(lightPos - world_pos);
    vec3 H = normalize(V + L);
    float distance = length(lightPos - world_pos);
    float attenuation = 1.0 / (distance * distance);
    //vec3 radiance = lightColor * attenuation;
    vec3 radiance = lightColor;
    vec3 F0 = vec3(0.12);
    F0 = mix(F0, material.color.xyz, material.mettalic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float NDF = DistributionGGX(N, H, material.roughness);
    float G = GeometrySmith(N, V, L, material.roughness);
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.mettalic;
    float NdotL = max(dot(N, L), 0.0);
    vec3 Lo = (kD * material.color.xyz / PI + specular) * radiance * NdotL;
    vec3 ambient = vec3(0.03) * material.color.xyz;
    vec3 color = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    f_color = vec4(3 * color, 1.0);
}
