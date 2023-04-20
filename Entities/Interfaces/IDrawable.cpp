#include "../../Services/Definitions.hpp"
#include "IDrawable.hpp"


IDrawable::IDrawable(Window &ConcreteWindow, Shader &ConcreteShader)
	: Window_(ConcreteWindow), Shader_(ConcreteShader) {
	
	Window_.reserveSimpleObject();
	VAOId_ = Window_.createVAO();
	VBOId_ = Window_.createVBO();
}

IDrawable::IDrawable(Window &ConcreteWindow)
	: Window_(ConcreteWindow), Shader_(Shader()) {}

void IDrawable::draw() {
	// ���������� ��������� ���������
	Shader_.use();
}

// �������� ���� ���-�� �������� �������, �� ��
IDrawable::~IDrawable() = default;