#pragma once

#include <vector>

class Params1D {
	std::vector<float> k{}; // значения наклонов на углах отрезков
	// numsteps_ = k.size
	float step{};
	uint32_t dx{};
	uint32_t controlPoints{};
	uint32_t octaveNum{};
	uint32_t resultDotsCols{};
};