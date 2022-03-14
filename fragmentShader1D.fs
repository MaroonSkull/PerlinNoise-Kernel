#version 330 core

in float pointColor;

out vec4 FragColor;

void main() {
	FragColor = vec4(pointColor, pointColor, pointColor, 1.0f);
}