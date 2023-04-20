#pragma once

class Background final : private IDrawable {
private:
	Params &p_;
public:
	Background(Window &ConcreteWindow, Params &p);
	void draw();
};