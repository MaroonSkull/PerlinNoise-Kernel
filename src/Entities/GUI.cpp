#include "../Services/Definitions.hpp"
#include "GUI.hpp"

GUI::GUI(Window &ConcreteWindow, Params &p)
	: p_(p), IDrawable(ConcreteWindow) {

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	
	ImFontConfig font_config;
	font_config.OversampleH = 1; //or 2 is the same
	font_config.OversampleV = 1;
	font_config.PixelSnapH = 1;

	static const ImWchar ranges[] =
	{
		0x0020, 0x00FF, // Basic Latin + Latin Supplement
		0x0400, 0x044F, // Cyrillic
		0,
	};

	io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\Tahoma.ttf", 16.0f, &font_config, ranges);
	
	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(ConcreteWindow.getWindow(), true);
	ImGui_ImplOpenGL3_Init();
}

GUI::~GUI() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void GUI::calc() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();


	ImGui::SetNextWindowPos(ImVec2(10, 10));
	ImGui::SetNextWindowSize(ImVec2(250, 270));
	ImGui::Begin(u8"����������");
	ImGui::Text(u8"��/���� � �������: %.3f [%.1f FPS]", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		
	ImGui::BulletText(u8"������: %d", p_.controlPoints_);
	if(ImGui::BeginTable("split", 2)) {
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.decrControlPoint();
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.incrControlPoint();
		ImGui::EndTable();
	}
	ImGui::BulletText(u8"����� �� ����������: %d", p_.numSteps_);
	if(ImGui::BeginTable("split1", 2)) {
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.decrNumSteps();
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.incrNumSteps();
		ImGui::EndTable();
	}
	ImGui::BulletText(u8"��������� �����: %d", p_.octaveNum_);
	if(ImGui::BeginTable("split2", 2)) {
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.decrOctave();
		ImGui::TableNextColumn();
		if(ImGui::Button(u8"���������"))
			p_.incrOctave();
		ImGui::EndTable();
	}

	ImGui::BulletText(u8"����� �����: %d", p_.resultDotsCols_);
	ImGui::BulletText(u8"dx ��� 2 �����: %.6f", p_.step_);
	ImGui::ColorEdit3(u8"���� ���� ", (float *)&p_.backgroundColor);
	ImGui::End();


	ImGui::SetNextWindowPos(ImVec2(270, 10));
	ImGui::SetNextWindowSize(ImVec2(220, 270));
	ImGui::Begin(u8"������� ������", NULL, ImGuiWindowFlags_AlwaysVerticalScrollbar);

	std::string ks = {"k[0]"};
	size_t i = 0;
	float both = p_.k_.at(0);
	if(i == p_.activeId_) ks = "->" + ks;
	ImGui::SliderFloat(ks.c_str(), &both, -1.0f, 1.0f);
	for(i = 1; i < p_.k_.size() - 1; i++) {
		ks = "k[" + std::to_string(i) + "]";
		if(i == p_.activeId_) ks = "->" + ks;
		ImGui::SliderFloat(ks.c_str(), &p_.k_.at(i), -1.0f, 1.0f);
	}
	ks = "k[" + std::to_string(i) + "]";
	if(i == p_.activeId_) ks = "->" + ks;
	ImGui::SliderFloat(ks.c_str(), &both, -1.0f, 1.0f);
	p_.k_.at(i) = both;
	p_.k_.at(0) = both;

	ImGui::End();


	ImGui::SetNextWindowPos(ImVec2(500, 10));
	ImGui::SetNextWindowSize(ImVec2(600, 270));
	ImGui::Begin(u8"�������", &showHelpSubwindow);

	ImGui::BulletText(u8"������� �������� ������ ��� ��������� ����� ��������� �����/������.");
	ImGui::BulletText(u8"�������� ������ �������� �������� � ���� �����.");
	ImGui::BulletText(u8"�������� ���� ������� �������� ������ �����, ��������� ��������� ���� ��� �������.");
	ImGui::BulletText(u8"������� ������ � ��������� ������ ������ �����.");
	ImGui::BulletText(u8"�������� ���������� ������� ����� ����� ��� ������ ������ �����/����.");
	ImGui::BulletText(u8"����� lCtrl, �������� ����� �������� �� 10 ������.");
	ImGui::BulletText(u8"����� lShift, �������� ����� �������� �� 100 ������.");
	ImGui::BulletText(u8"�������� ���������� ����� �����, ��������� ������� W/S.");
	ImGui::BulletText(u8"�������� ���������(���������� �� ���� ����������) ����� ����� ��� ������ ������ A/D.");
	ImGui::End();
}

void GUI::draw() {
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
