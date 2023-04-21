#version 330 core
layout (location = 0)

in vec2 aPos;

out vec3 pointColor;

uniform bool monoColor_;
uniform vec4 color_;

void main() {
	gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
	if (monoColor_)
		pointColor = color_.rgb;
	else {
		float color = (aPos.y+1)/2;
		
		if (aPos.y > 0.0)
			pointColor = vec3(color, 1.0-color, 0.0);
		else
			pointColor = vec3(0.0, 1.0-color, color);
	}
}