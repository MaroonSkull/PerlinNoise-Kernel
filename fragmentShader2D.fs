#version 330 core

in float pointColor;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;

void main() {
	FragColor = texture(ourTexture, TexCoord);
}