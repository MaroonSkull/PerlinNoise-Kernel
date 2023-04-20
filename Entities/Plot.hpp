#pragma once


class Plot final : private IDrawable, private ICalculable {
private:
	Params &p_;
	std::vector<float> ks_;
public:
	Plot(Window &ConcreteWindow, Shader &ConcreteShader, Params &p);

	void calc();
	void draw();
};

