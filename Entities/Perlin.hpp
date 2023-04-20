#pragma once


class Perlin final : private IDrawable, private ICalculable {
private:
	Params &p_;
	std::vector<float> vertices_;
public:
	Perlin(Window &ConcreteWindow, Shader &ConcreteShader, Params &p);

	void calc();
	void draw();
};
