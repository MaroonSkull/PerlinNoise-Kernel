#include "../Services/Definitions.hpp"
#include "Plot.hpp"

Plot::Plot(Window &ConcreteWindow, Shader &ConcreteShader, Params &p)
	: p_(p), IDrawable(ConcreteWindow, ConcreteShader) {
	
	ks_ = {
		-1.0, 0,
		1.0, 0,
		0, -1,
		0, 1,
	}; // x, y на каждую из 2*controlPoints точек для прямых + 8 точек для осей OXY.
}

void Plot::calc() {
	ks_.resize(8 + 2 * 2 * p_.controlPoints_);
	for(int i = 0; i < p_.controlPoints_; i++) {
		float x0 = static_cast<float>(i) / (p_.controlPoints_ - 1);
		float deltaX = 1.f / (p_.controlPoints_ - 1) / 4;

		ks_.at(8 + 4 * i) = 2 * (x0 - deltaX) - 1;						// x left
		ks_.at(8 + 4 * i + 1) = (p_.controlPoints_ - 1) * p_.k_.at(i) * -deltaX;	// y left
		ks_.at(8 + 4 * i + 2) = 2 * (x0 + deltaX) - 1;					// x right
		ks_.at(8 + 4 * i + 3) = (p_.controlPoints_ - 1) * p_.k_.at(i) * deltaX;	// y right
	}
	Window_.bindFillBuffer(VAOId_, VBOId_, ks_);
}

void Plot::draw() {
	IDrawable::draw();
	
	glLineWidth(1.0);
	
	glBindVertexArray(Window_.getVAO(VAOId_));
	
	for(int i = 0; i < p_.controlPoints_ + 4; i++) {
		glDrawArrays(GL_LINE_STRIP, 2 * i, 2);
	}
	glBindVertexArray(0);
}