#pragma once

namespace cudaWrp {
	class ErrorException {
	private:
		mutable std::string e_;
		Params p;
	public:
		ErrorException(cudaError_t status, std::string_view name, const Params &p);
		ErrorException(cudaError_t status, std::string_view name);
		std::string getError() const;
	};
}