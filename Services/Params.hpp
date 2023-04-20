#pragma once

#include <vector>
#include <functional>
#include <imgui_impl_opengl3.h>

class Params {
private:
	// функция обязана возвращать значения в промежутке [-1, 1]
	std::function<float(void)> vectorGenerator_;

	void updateResultDotsCols();
	void updateStep();
public:
	ImVec4 backgroundColor = ImVec4(0.2f, 0.3f, 0.3f, 1.00f);
	int32_t activeId_ = 0;
	uint32_t dx_ = 1;
	uint32_t controlPoints_ = 0;
	int32_t numSteps_ = 0;
	uint32_t octaveNum_ = 0;
	uint32_t resultDotsCols_ = 0;
	float step_ = 0.f;
	std::vector<float> k_; // значения наклонов на углах отрезков

	Params(std::function<float(void)> vectorGenerator, uint32_t controlPoints, uint32_t numSteps, uint32_t octaveNum);
	Params();

	void incrControlPoint();
	void decrControlPoint();
	void incrActive();
	void decrActive();
	void incrOctave();
	void decrOctave();
	void incrNumSteps();
	void decrNumSteps();
};