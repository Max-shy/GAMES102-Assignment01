#pragma once

#include <UGM/UGM.h>
#include"imgui/imgui.h"


struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	//插值结果
	std::vector<ImVec2> Lagrange_Result;//拉格朗日多项式插值结果（点）
	std::vector<ImVec2> Gauss_Result;//高斯基函数线性插值结果（点）
	
	//回归结果
	std::vector<ImVec2> LS_Result;//最小二乘法回归结果（点）
	std::vector<ImVec2> Ridge_Regression_Result;//岭回归拟合结果（点）


	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };//网格开关
	bool opt_enable_context_menu{ true };//右键菜单栏开关
	bool adding_line{ false };

	bool opt_lagrange{ false };//拉格朗日插值 开关
	bool opt_Gauss{ false };//高斯基函数插值 开关
	bool opt_LS{ false };//最小二乘法回归 开关
	bool opt_Ridge_Regression{ false };// 岭回归拟合 开关

	//参数
	int LeastSquaresM = 4;
	float RidgeRegressionLambda = 0.1;
	float GaussTheta = 10;
	
};

#include "details/CanvasData_AutoRefl.inl"
