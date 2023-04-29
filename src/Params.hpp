#pragma once

#include <vector>

struct Params1D {
	// numsteps_ = k.size
	float step{};
	uint32_t dx{};
	uint32_t controlPoints{};
	uint32_t octaveNum{};
	uint32_t resultDotsCols{};
	std::vector<float> k{}; // value of slopes at the corners of segments
};