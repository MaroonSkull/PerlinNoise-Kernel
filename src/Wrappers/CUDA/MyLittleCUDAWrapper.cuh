#pragma once

#include <cuda_runtime.h>
#include "../../Services/Params.hpp"
#include <sstream>
#include <functional>

namespace cudaWrp {
	class ErrorException;

	void callAPI(cudaError_t status, std::string_view name);
	void callAPI(cudaError_t status, std::string_view name, const Params &p);
	cudaError_t callKernel(std::function<void(void)>kernel, std::string_view name, const Params &p, bool synchronize = true);

	void initializeContext();
	void destroyContext();
}