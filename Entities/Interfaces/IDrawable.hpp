#pragma once


class IDrawable {
protected:
	Window &Window_;
	Shader &Shader_;
	size_t VAOId_ = 0;
	size_t VBOId_ = 0;

	IDrawable(Window &ConcreteWindow, Shader &ConcreteShader);
	IDrawable(Window &ConcreteWindow);
public:
	virtual ~IDrawable();

	virtual void draw();
};
