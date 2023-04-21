#include "MyLittleCUDAWrapper.cuh"
#include "ErrorException.hpp"



void
cudaWrp::callAPI(cudaError_t status, std::string_view name) {
	if(status != cudaSuccess) {
		throw cudaWrp::ErrorException(status, name);
	}
}

void
cudaWrp::callAPI(cudaError_t status, std::string_view name, const Params &p) {
	if(status != cudaSuccess) {
		throw cudaWrp::ErrorException(status, name, p);
	}
}

cudaError_t
cudaWrp::callKernel(std::function<void(void)>kernel, std::string_view name, const Params &p, bool synchronize) {
	cudaError_t status = cudaError::cudaErrorUnknown;

	kernel();

	if(synchronize) {
		// Check for any errors launching the kernel
		callAPI(status = cudaGetLastError(), "cudaGetLastError", p);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		callAPI(status = cudaDeviceSynchronize(), "cudaDeviceSynchronize", p);
	}

	return status;
}

//must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
void cudaWrp::destroyContext() {
	callAPI(cudaDeviceReset(), "cudaDeviceReset");
}
