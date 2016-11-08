#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 surfaceNormal;
layout(location = 3) in vec3 toLightVector;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 unitNormal = normalize(surfaceNormal); 
    vec3 unitLightVector = normalize(toLightVector);
	
    float bright = max(dot(unitNormal,unitLightVector),0.1f);
    vec3 diffuse = bright*vec3(1.0,1.0,1.0);

    outColor = vec4(diffuse,1.0)*texture(texSampler,fragTexCoord);
    //outColor = vec4(fragColor,1.0);
}