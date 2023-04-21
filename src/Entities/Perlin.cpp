#include "../Services/Definitions.hpp"
#include "../CUDAKernels/Kernels.cuh"
#include "Perlin.hpp"


Perlin::Perlin(Window &ConcreteWindow, Shader &ConcreteShader, Params &p)
	: p_(p), IDrawable(ConcreteWindow, ConcreteShader) {

	cudaWrp::initializeContext();
}

void Perlin::calc() {
	vertices_.resize(2 * (p_.resultDotsCols_ + 1)); //x, y to 1 dot -> size = 2*cols

	// Calculate Perlin in parallel.
	Perlin1DWithCuda<float>(vertices_.data(), p_); // can throw exception
	
	Window_.bindFillBuffer(VAOId_, VBOId_, vertices_);
}

void Perlin::draw() {
	IDrawable::draw();

	glLineWidth(2);
	// предоставляем выполнение бинда gl
	glBindVertexArray(Window_.getVAO(VAOId_));

	// Рисуем шум Перлина
	glDrawArrays(GL_LINE_STRIP, 0, p_.resultDotsCols_+1);
}