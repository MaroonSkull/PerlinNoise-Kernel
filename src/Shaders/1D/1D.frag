#version 330 core

in vec3 pointColor;

out vec4 FragColor;

void main() {
	FragColor = vec4(pointColor.r, pointColor.g, pointColor.b, 1.0f);
}