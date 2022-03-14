#pragma once
class Shader {
private:
    bool readFromFile(std::string *accumulator, const GLchar *pathToFile);
    // the program ID
    uint32_t shaderId = 0;

public:
    // constructor reads and builds the shader
    Shader(const char *vertexShaderPath, const char *fragmentShaderPath);
    ~Shader();

    // use/activate the shader
    void use();
    
    // utility uniform functions
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int32_t value) const;
    void setFloat(const std::string &name, float value) const;
    
    // getters/setters
    int getShaderId() const;
    void setShaderId(uint32_t shaderId);
};