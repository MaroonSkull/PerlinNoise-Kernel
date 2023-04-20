#pragma once


class ICalculable {
protected:
	ICalculable() {};
public:
	virtual ~ICalculable() = default;

	virtual void calc() = 0;
};
