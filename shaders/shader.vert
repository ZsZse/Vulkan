#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 colorMixer;
    vec3 lightPosition;
} ubo;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 surfaceNormal;
layout(location = 3) out vec3 toLightVector;

void main() {
    vec4 worldPos = ubo.model*vec4(inPosition, 1.0);
    gl_Position = ubo.proj*ubo.view*worldPos;
    fragColor = vec3(inColor.x*ubo.colorMixer.x,inColor.y*ubo.colorMixer.y,inColor.z*ubo.colorMixer.z);
    fragTexCoord = inTexCoord;
	
    surfaceNormal=(ubo.model*vec4(inNormal,0.0)).xyz;
    toLightVector=ubo.lightPosition-worldPos.xyz; // (ubo.view*vec4(inPosition,1.0)).xyz
}