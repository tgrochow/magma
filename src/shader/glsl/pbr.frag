#version 450

const float PI = 3.14159265359;

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform CameraData {
    vec3 pos;
} camera;

struct PointLight {
    vec4 pos;
    vec4 color;
};

struct DirLight {
    vec4 dir;
    vec4 color;
};

layout(std430, set = 0, binding = 2) readonly buffer PointLightBuffer
{
    PointLight lights[];
} pointLights;

layout(std430, set = 0, binding = 3) readonly buffer DirLightBuffer
{
    DirLight lights[];
} dirLights;

layout(push_constant) uniform PushConstantData {
    vec4 materialColor;
    float materialMettalic;
    float materialRoughness;
    uint pointLightCount;
    uint dirLightCount;
} params;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness)
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

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 calcColor(vec3 N, vec3 V, vec3 L, float attenuation, vec3 lightColor) {
    vec3 H = normalize(V + L);
    vec3 radiance = lightColor * attenuation;
    //vec3 radiance = lightColor;
    vec3 F0 = vec3(0.12);
    F0 = mix(F0, params.materialColor.xyz, params.materialMettalic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float NDF = distributionGGX(N, H, params.materialRoughness);
    float G = geometrySmith(N, V, L, params.materialRoughness);
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - params.materialMettalic;
    float NdotL = max(dot(N, L), 0.0);
    vec3 Lo = (kD * params.materialColor.xyz / PI + specular) * radiance * NdotL;
    vec3 ambient = vec3(0.03) * params.materialColor.xyz;
    vec3 color = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    return color;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera.pos - world_pos);
    vec3 accColor = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < params.pointLightCount; ++i) {
        vec3 lightPos = pointLights.lights[i].pos.xyz;
        vec3 lightColor = pointLights.lights[i].color.xyz;
        vec3 L = normalize(lightPos - world_pos);
        float distance = length(lightPos - world_pos);
        float attenuation = 1.0 / (distance * distance);
        accColor += calcColor(N, V, L, attenuation, lightColor);
    }
    for (int i = 0; i < params.dirLightCount; ++i) {
        vec3 lightDir = dirLights.lights[i].dir.xyz;
        vec3 lightColor =  dirLights.lights[i].color.w * dirLights.lights[i].color.xyz;
        vec3 L = normalize(-lightDir);
        float attenuation = 1.0;
        accColor += calcColor(N, V, L, attenuation, lightColor);
    }
    f_color = vec4(accColor, 1.0);
}
