# GAMES102-Assignment01
GAMES102作业1，通过插值、回归完成曲线拟合

## Assignment 01

Assignment 01 required curve fitting to be completed by using 4 different methods. Including 2 interpolation fittings and 2 regression fittings.

###  Interpolation fitting 

Interpolation fitting mainly realizes Lagrange polynomial interpolation and Gaussian basis function interpolation. Polynomial interpolation is accomplished with different basis functions.

- **Lagrange polynomial interpolation**
- 
![image-20220612101310800](https://user-images.githubusercontent.com/68177870/173277956-fe2de206-9772-4547-a384-7623177f3d85.png)

  Complete the code according to the formula.

  ```cpp
  //拉格朗日插值中的基函数
  float Pk(const std::vector<Ubpa::pointf2>& points, int k, float x) {
  	int n = points.size();//点的数量
  	float pk = 1;
  	for (int i = 0; i < n; i++) {
  		if (i == k) continue;
  		pk = pk * (x - points[i][0]) / (points[k][0] - points[i][0]);
  	}
  	return pk;
  }
  
  //拉格朗日插值多项式
  float Polynomial(const std::vector<Ubpa::pointf2>& points, float x) {
  	int n = points.size();
  	float ans = 0;
  	for (int i = 0; i < n; i++) {
  		ans += points[i][1] * Pk(points, i, x);
  	}
  	return ans;
  }
  ```
  
  ![image-20220612101714760](https://user-images.githubusercontent.com/68177870/173277983-a2ada69d-b234-4b09-a855-2bf717dd45c2.png)


- **Gaussian basis function interpolation**

  Gaussian basis function:

![image-20220612165405855](https://user-images.githubusercontent.com/68177870/173278007-613c5748-f018-4443-83fb-3690403aebc5.png)

  Polynomial interpolation:

![image-20220612102855479](https://user-images.githubusercontent.com/68177870/173278028-5a644608-c568-432b-a6b3-64fe69ef6d5b.png)

  Complete the code according to the formula.

  ```cpp
  //Gauss基函数线性插值
  float Gauss(const std::vector<Ubpa::pointf2>& points, float x,float theta) {
  	int n = points.size();
  	if (n == 0) return 0;
  	//float theta = 100;
  
  	Eigen::MatrixXf A(n, n);//g(x)矩阵A计算
  	for (int i = 0; i < n; i++) {
  		for (int j = 0; j < n; j++) {
  			A(i, j) = (std::exp(-(points[i][0] - points[j][0]) * (points[i][0] - points[j][0]) / (2 * theta * theta)));
  		}
  	}
  	Eigen::VectorXf y(n);
  	for (int i = 0; i < n; i++) {
  		y(i) = points[i][1];
  	}
  
  	//计算Ab=y中的a(向量)
  	Eigen::VectorXf b = A.colPivHouseholderQr().solve(y);
  	float ans = 0;
  	for (int i = 0; i < n; i++) {
  		ans += b[i] * (std::exp(-(x - points[i][0]) * (x - points[i][0]) / (2 * theta * theta)));
  	}
  	return ans;
  }
  ```

  Observe the change in the curve by modulating theta.

  theta = 10:

  ![image-20220612103344263](https://user-images.githubusercontent.com/68177870/173278043-c86aecca-d824-4507-a540-8e54328a1a28.png)
  

  theta = 50:
  

  theta = 100:
  

### Regression fitting

Regression fitting mainly realizes the least square regression and ridge regression, which use the power basis function as the polynomial function. 

- **Least Square regression**

  Fix the highest power m of a polynomial function and minimize the error of the function.

  <img src="E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612105201136.png" alt="image-20220612105201136" style="zoom:80%;" />

  <img src="E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612105223088.png" alt="image-20220612105223088" style="zoom:67%;" />

  The polynomial coefficients can be obtained by solving the linear equations.

  <img src="E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612105411107.png" alt="image-20220612105411107" style="zoom:67%;" />

  ```CPP
  //最小二乘法回归
  float LS(const std::vector<Ubpa::pointf2>& points, float x, int m) {
  	int n = points.size();
  	if (n == 0) return 0;
  	if (n <= m) m = n - 1;//最高次幂不能大于点数
  
  	Eigen::MatrixXf X(n, m+1);//幂矩阵
  	for (int i = 0; i < n; i++) {
  		for (int j = 0; j <= m; j++) {
  			X(i, j) = std::powf(points[i][0], j);
  		}
  	}
  	Eigen::VectorXf Y(n);
  	for (int i = 0; i < n; i++) {
  		Y(i) = points[i][1];
  	}
  	
  	//求系数
  	Eigen::VectorXf Theta = (X.transpose() * X).inverse() * (X.transpose() * Y);
  	float ans = 0;
  	for (int i = 0; i <= m; i++)
  	{
  		ans += Theta[i] * std::powf(x, i);
  	}
  	return ans;
  }
  ```

Observe the change in the curve by modulating m.

m=3:

![image-20220612111033293](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111033293.png)

m=4:

![image-20220612111111831](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111111831.png)



- **Ridge regression**

  Ridge regression is to add regular term lambda based on the least square method.

  ![image-20220612111407396](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111407396.png)

  The polynomial coefficients can be obtained by solving the linear equations.

  ![image-20220612111415717](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111415717.png)

  ```CPP
  //岭回归
  float Ridge_Regression(const std::vector<Ubpa::pointf2>& points,float x, float lambda,int m) {
  	int n = points.size();
  	if (n == 0)return 0;
  	if (n <= m) m = n - 1;
  	
  	Eigen::MatrixXf X(n, m+1);
  	for (int i = 0; i < n; i++) {
  		for (int j = 0; j <= m; j++) {
  			X(i, j) = std::powf(points[i][0], j);
  		}
  	}
  
  	Eigen::VectorXf Y(n);
  	for (int i = 0; i < n; i++) {
  		Y(i) = points[i][1];
  	}
  	Eigen::MatrixXf I(m+1, m+1);
  	I.setIdentity();
  	Eigen::VectorXf Theta = (X.transpose() * X + I * lambda).inverse() *( X.transpose() * Y);
  	float ans = 0;
  	for (int i = 0; i <= m; i++) {
  		ans += Theta[i] * std::powf(x, i);
  	}
  	return ans;
  }
  ```

  Observe the change in the curve by modulating m and lambda.

  m=4，lambda=0.001:

  ![image-20220612111626596](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111626596.png)

  

  m=3，lambda=0.001:

  ![image-20220612111715606](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612111715606.png)



Now, compare the four different fitting results.

![image-20220612112316714](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612112316714.png)

![image-20220612112553239](E:\CG\Games\GAMES102\Study report\Week1\Study report for week 3.assets\image-20220612112553239.png)

code: [Max-shy/GAMES102-Assignment01](https://github.com/Max-shy/GAMES102-Assignment01)

