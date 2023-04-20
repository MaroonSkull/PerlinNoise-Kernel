#include <algorithm>
#include "Params.hpp"


void Params::updateResultDotsCols() {
	resultDotsCols_ = (controlPoints_ - 1) * numSteps_;
}

void Params::updateStep() {
	step_ = 1.0f / numSteps_;
}

Params::Params(std::function<float(void)> vectorGenerator, uint32_t controlPoints, uint32_t numSteps, uint32_t octaveNum)
	: vectorGenerator_(vectorGenerator), controlPoints_(controlPoints), numSteps_(numSteps), octaveNum_(octaveNum) {

	updateResultDotsCols();
	updateStep();

	k_.reserve(controlPoints_);
	for(int i = 0; i < controlPoints_ - 1; i++) {
		k_.push_back(vectorGenerator_());
	}
	k_.push_back(k_[0]); // (последний наклон равен первому)
}

Params::Params() {}

void Params::incrControlPoint() {
	k_.back() = vectorGenerator_();
	k_.push_back(k_.at(0));

	controlPoints_++;
	resultDotsCols_ += numSteps_;
}

void Params::decrControlPoint() {
	if(k_.size() <= 2)
		return;

	k_.pop_back();
	k_.back() = k_.at(0);

	resultDotsCols_ -= numSteps_;
	controlPoints_--;
	if(activeId_ == k_.size())
		activeId_--;
}

void Params::incrActive() {
	activeId_++;
	if(activeId_ >= controlPoints_)
		activeId_ = 1;
}

void Params::decrActive() {
	activeId_--;
	if(activeId_ < 0)
		activeId_ = controlPoints_ - 2;
}

void Params::incrOctave() {
	if(numSteps_ * controlPoints_ / (1 << octaveNum_))
		octaveNum_++;
}

void Params::decrOctave() {
	if(octaveNum_ < 1)
		return;
	octaveNum_--;
}

void Params::incrNumSteps() {
	numSteps_ += dx_;
	updateResultDotsCols();
	updateStep();
}

void Params::decrNumSteps() {
	numSteps_ -= dx_;
	if(numSteps_ < 2)
		numSteps_ = 2;
	updateResultDotsCols();
	updateStep();
}