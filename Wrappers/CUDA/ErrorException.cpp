#include "../../Services/Params.hpp"
#include "MyLittleCUDAWrapper.cuh"
#include "ErrorException.hpp"
#include <iomanip>


using namespace cudaWrp;


cudaWrp::ErrorException::ErrorException(cudaError_t status, std::string_view name, const Params &p) {
	
	std::stringstream ss;
	
	ss << std::endl << std::endl << "****************** CUDA exception catched ******************" << std::endl << std::endl
		<< name << " throwed " << cudaGetErrorName(status) << std::endl
		<< cudaGetErrorString(status) << std::endl << std::endl

		<< std::left << std::setw(14) << "********* Params *********" << std::endl
		<< std::left << std::setw(14) << "activeId" << " = " << p.activeId_ << std::endl
		<< std::left << std::setw(14) << "dx" << " = " << p.dx_ << std::endl
		<< std::left << std::setw(14) << "controlPoints" << " = " << p.controlPoints_ << std::endl
		<< std::left << std::setw(14) << "numSteps" << " = " << p.numSteps_ << std::endl
		<< std::left << std::setw(14) << "octaveNum" << " = " << p.octaveNum_ << std::endl
		<< std::left << std::setw(14) << "resultDotsCols" << " = " << p.resultDotsCols_ << std::endl
		<< std::left << std::setw(14) << "step" << " = " << p.step_ << std::endl << std::endl;

	ss << std::left << std::setw(18) << "************ k ***********" << std::endl;
	for(int i = 0; i < p.k_.size(); i++)
		ss << std::left << "k[" << std::setw(std::to_string(p.controlPoints_-1).size()) << i << "] = " << std::setprecision(8) << p.k_.at(i) << std::endl;

	ss << std::endl << "************************************************************" << std::endl << std::endl;
	
	e_ = ss.str();
}

ErrorException::ErrorException(cudaError_t status, std::string_view name) {
	std::stringstream ss;
	ss << std::endl << std::endl << "****************** CUDA exception catched ******************" << std::endl
		<< name << " throwed " << cudaGetErrorName(status) << std::endl
		<< cudaGetErrorString(status) << std::endl
		<< "*************************************************************" << std::endl << std::endl;
	e_ = ss.str();
}

std::string ErrorException::getError() const {
	return e_;
}