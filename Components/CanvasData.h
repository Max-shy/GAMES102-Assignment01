#pragma once

#include <UGM/UGM.h>
#include"imgui/imgui.h"


struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	//��ֵ���
	std::vector<ImVec2> Lagrange_Result;//�������ն���ʽ��ֵ������㣩
	std::vector<ImVec2> Gauss_Result;//��˹���������Բ�ֵ������㣩
	
	//�ع���
	std::vector<ImVec2> LS_Result;//��С���˷��ع������㣩
	std::vector<ImVec2> Ridge_Regression_Result;//��ع���Ͻ�����㣩


	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };//���񿪹�
	bool opt_enable_context_menu{ true };//�Ҽ��˵�������
	bool adding_line{ false };

	bool opt_lagrange{ false };//�������ղ�ֵ ����
	bool opt_Gauss{ false };//��˹��������ֵ ����
	bool opt_LS{ false };//��С���˷��ع� ����
	bool opt_Ridge_Regression{ false };// ��ع���� ����

	//����
	int LeastSquaresM = 4;
	float RidgeRegressionLambda = 0.1;
	float GaussTheta = 10;
	
};

#include "details/CanvasData_AutoRefl.inl"
