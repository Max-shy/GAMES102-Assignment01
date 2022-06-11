#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>
#include"../Eigen/Core"
#include"../Eigen/Dense"


using namespace Ubpa;

//�������ղ�ֵ�еĲ�ֵ����
float Pk(const std::vector<Ubpa::pointf2>& points, int k, float x) {
	int n = points.size();//�������
	float pk = 1;
	for (int i = 0; i < n; i++) {
		if (i == k) continue;
		pk = pk * (x - points[i][0]) / (points[k][0] - points[i][0]);
	}
	return pk;
}

//�������ն���ʽ��ֵ
float Polynomial(const std::vector<Ubpa::pointf2>& points, float x) {
	int n = points.size();
	float ans = 0;
	for (int i = 0; i < n; i++) {
		ans += points[i][1] * Pk(points, i, x);
	}
	return ans;
}

//Gauss���������Բ�ֵ
float Gauss(const std::vector<Ubpa::pointf2>& points, float x,float theta) {
	int n = points.size();
	if (n == 0) return 0;
	//float theta = 100;

	Eigen::MatrixXf A(n, n);//g(x)����A����
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A(i, j) = (std::exp(-(points[i][0] - points[j][0]) * (points[i][0] - points[j][0]) / (2 * theta * theta)));
		}
	}
	Eigen::VectorXf y(n);
	for (int i = 0; i < n; i++) {
		y(i) = points[i][1];
	}

	//����Ax=y�е�x(����)
	Eigen::VectorXf a = A.colPivHouseholderQr().solve(y);
	float ans = 0;
	for (int i = 0; i < n; i++) {
		ans += a[i] * (std::exp(-(x - points[i][0]) * (x - points[i][0]) / (2 * theta * theta)));
	}
	return ans;
}

//��С���˷��ع�
float LS(const std::vector<Ubpa::pointf2>& points, float x, int m) {
	int n = points.size();
	if (n == 0) return 0;
	if (n <= m) m = n - 1;//��ߴ��ݲ��ܴ��ڵ���

	Eigen::MatrixXf X(n, m);//�ݾ���
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			X(i, j) = std::powf(points[i][0], j);
		}
	}

	Eigen::VectorXf Y(n);
	for (int i = 0; i < n; i++) {
		Y(i) = points[i][1];
	}
	
	//��ϵ��
	Eigen::VectorXf Theta = (X.transpose() * X).inverse() * X.transpose() * Y;
	float ans = 0;
	for (int i = 0; i < m; i++)
	{
		ans += Theta[i] * std::powf(x, i);
	}
	return ans;
}

//��ع�
float Ridge_Regression(const std::vector<Ubpa::pointf2>& points,float x, float lambda,int m) {
	int n = points.size();
	if (n == 0)return 0;
	if (n <= m) m = n - 1;
	
	Eigen::MatrixXf X(n, m);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			X(i, j) = std::powf(points[i][0], j);
		}
	}

	Eigen::VectorXf Y(n);
	for (int i = 0; i < n; i++) {
		Y(i) = points[i][1];
	}
	Eigen::MatrixXf I(m, m);
	I.setIdentity();
	Eigen::VectorXf Theta = (X.transpose() * X + I * lambda).inverse() * X.transpose() * Y;
	float ans = 0;
	for (int i = 0; i < m; i++) {
		ans += Theta[i] * std::powf(x, i);
	}
	return ans;
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);//�Ƿ��������
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);//�Ƿ������Ҽ��ı���
			ImGui::Text("Mouse Left: drag to add point,\nMouse Right: drag to scroll, click for context menu.");//�ı�

			//��ֵ����
			ImGui::Checkbox("Lagrange", &data->opt_lagrange);//�Ƿ�����������ղ�ֵ
			ImGui::Checkbox("Gauss", &data->opt_Gauss);//�Ƿ���и�˹��������ֵ
			ImGui::SameLine(200);
			ImGui::InputFloat("theta", &data->GaussTheta);//��˹�ƺ���theta

			//�ع鿪��
			ImGui::Checkbox("LS", &data->opt_LS);//�Ƿ������С���˷��ع�
			ImGui::SameLine(200);
			ImGui::InputInt("m", &data->LeastSquaresM);//������С���˷����ݻ�����ߴ���

			ImGui::Checkbox("Ridge_Regression", &data->opt_Ridge_Regression);//�Ƿ������ع�
			ImGui::SameLine(200);
			ImGui::InputFloat("lambda", &data->RidgeRegressionLambda);//��ع�lambda

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      //ImDrawList APIʹ����Ļ����!  
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   //���������Ĵ�СΪ���õ�  
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// ���Ʊ߿�ͱ�����ɫ
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// �⽫��׽�����ǵĻ���
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); //ԭ��
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);//����ĵ�����

			//��������
			if (is_hovered /*&& !data->adding_line*/ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);//�������м��������
				
				//ȷ������x�᷶Χ
				float Xmin = 999999;
				float Xmax = -99999;
				for (size_t i = 0; i < data->points.size(); i++) {
					if (data->points[i][0] < Xmin){
						Xmin = data->points[i][0];
					}
					if (data->points[i][0] > Xmax){
						Xmax = data->points[i][0];
					}
				}
				//��ʼ����ֵ/�ع����ߵ������
				data->Lagrange_Result.clear();
				data->Gauss_Result.clear();
				data->LS_Result.clear();
				data->Ridge_Regression_Result.clear();
				
				//�����ֵ/�ع���
				for (int x = Xmin - 1; x < Xmax + 2; x++) {
					data->Lagrange_Result.push_back(ImVec2(origin.x + x , origin.y + Polynomial(data->points, x)));
					data->Gauss_Result.push_back(ImVec2(origin.x + x, origin.y + Gauss(data->points, x,data->GaussTheta)));
					data->LS_Result.push_back(ImVec2(origin.x + x, origin.y + LS(data->points, x, data->LeastSquaresM)));
					data->Ridge_Regression_Result.push_back(ImVec2(origin.x + x, origin.y + Ridge_Regression(data->points, x, data->RidgeRegressionLambda, data->LeastSquaresM)));
				}
			}

			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;

			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				//if (data->adding_line)
				//	data->points.resize(data->points.size() - 2);
				//data->adding_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 2); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			//����
			for (int i = 0; i < data->points.size(); i++) {
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[i][0], origin.y + data->points[i][1]), 4.0f, IM_COL32(255, 255, 0, 255));
			}
			//for (int n = 0; n < data->points.size(); n += 2)
			//	draw_list->AddLine(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), IM_COL32(255, 255, 0, 255), 2.0f);

			//����
			if (data->opt_lagrange) {
				draw_list->AddPolyline(data->Lagrange_Result.data(), data->Lagrange_Result.size(), IM_COL32(64, 128, 255, 255), false, 1.0f);
			}
			if (data->opt_Gauss) {
				draw_list->AddPolyline(data->Gauss_Result.data(), data->Gauss_Result.size(), IM_COL32(128, 255, 255, 255), false, 1.0f);
			}
			if (data->opt_LS) {
				draw_list->AddPolyline(data->LS_Result.data(),data->LS_Result.size(), IM_COL32(255, 128, 128, 255), false, 1.0f);
			}
			if (data->opt_Ridge_Regression) {
				draw_list->AddPolyline(data->Ridge_Regression_Result.data(),data->Ridge_Regression_Result.size(), IM_COL32(255, 64, 64, 255), false, 1.0f);
			}
			
			draw_list->PopClipRect();
		}
		
		ImGui::End();
	});
}

