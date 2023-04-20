#pragma once

class GUI final: private IDrawable, private ICalculable {
private:
	Params &p_;
	bool showHelpSubwindow = false;
	bool showSlidersSubwindow = true;
	
public:
	GUI(Window &ConcreteWindow, Params &p);
	~GUI();

	void calc();
	void draw();
};
