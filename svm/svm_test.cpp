#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include "svm.h"

#define DATATYPE int
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace cv;
using namespace std;

template <class myType>
int LoadData(string fileName, cv::Mat& matData, int matRows, int matCols, int matChns);

template <class myType>
myType absolute(myType);

// Function to delete a row in opencv Mat
// @int  The row index (0 based)
Mat deleteRow(Mat, int);

// Funciton to check the similarity of two Mats of centroids
float check_similarity(Mat, Mat, float);

// Function to calculate the svm possibility
float poss(float);

// Function to transform Mat to vector<vector<>>
template <class myType>
vector<vector<myType>> matToVector_2d(Mat &);

template <class myType>
vector<myType> matToVector_1d(Mat &);

// Function to convert 2d vector to processed data for libsvm
template <class dataType, class labelType>
void process_Train_Data(int sizeofProblem, int elements, vector<vector<dataType>>& data, vector<labelType>& labels, svm_problem &prob, svm_node * &x_space);

// Function to convert 2d vector to processed data for libsvm (Test data)
template <class dataType>
void process_Test_Data(int sizeOfProblem, int elements, vector<vector<dataType>> &data, svm_node * &x_space);

// Function to create svm_node
template <class myType>
void getSvmNomde(int width, Mat &m, svm_node* node);

// Funciton to move particle around and get the highest probability image coordinates
template <class coodType, class probType>
pair<pair<coodType, coodType>, probType> move_particle(Mat &m, int x, int y, svm_model *model, int radius,int part_width, svm_node *& node); 

int main()
{
	Mat train_data;
	Mat label_data;
	Mat test_data;
	Mat testLabel_data;
	Mat test_centroids;

	int train_length = 6136;
	int test_length = 1535;

	int column = 400;
	int channel = 1;

	string inputData = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\train_data.txt";
    string inputLabels = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\labels_data.txt";
	
	string inputTestData = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\test_data.txt";
	string inputTestLabels = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\testLabel_data.txt";
	
	string inputCentroids = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\test_centroids.txt";
	// Data for visual representation
	cout<<"read data"<<LoadData<float>(inputData, train_data, train_length, column, channel)<<endl;
	cout<<"read labels"<<LoadData<float>(inputLabels, label_data, train_length, 1, channel)<<endl;
	cout<<"read TestData"<<LoadData<float>(inputTestData, test_data, test_length, column, channel)<<endl;
	cout<<"read TestLabels"<<LoadData<float>(inputTestLabels, testLabel_data, test_length, 1, channel)<<endl;
	cout<<"read testCentroids"<<LoadData<float>(inputCentroids, test_centroids, 500, 2, channel)<<endl;

	vector<vector<float>> aa = matToVector_2d<float>(test_centroids);
	
	cout<<"aa.size is: "<<aa.size()<<endl;

	//for(size_t i = 0;i < aa.size();i++){
	//	//for(size_t j = 0; j < aa[0].size();j++){
	//		cout<<aa[i][0]<<" "<<aa[i][1]<<endl;
	//	//}
	//}

	int train_rows = train_data.rows;
	int train_cols = train_data.cols;

	cout<<"inputData shape is: "<<train_rows<<" "<<train_cols<<endl;
	
	Size s = test_data.size();
	int test_rows = s.height;
	int test_cols = s.width;
	cout<<"inputLabels shape is: "<<test_rows<<" "<<test_cols<<endl;	

	struct svm_problem prob;
	struct svm_problem prob_test;
	struct svm_node *test_x_space = NULL;
	struct svm_node *x_space = NULL;
	struct svm_parameter param;
	struct svm_model *model;

	// Set up the svm parameters
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.00001;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-6;
	param.p = 0;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	vector<vector<float>> train_vector = matToVector_2d<float>(train_data);
	vector<float> train_label_vector = matToVector_1d<float>(label_data);

	vector<vector<float>> test_vector = matToVector_2d<float>(train_data);
	vector<float> test_label_vector = matToVector_1d<float>(label_data);

	process_Test_Data<float>(test_rows, test_cols, test_vector, test_x_space);
	
	const char *model_file_name = "./svm_model.txt";

	if(svm_load_model(model_file_name) == NULL){
		process_Train_Data<float, float>(train_rows, train_cols, train_vector, train_label_vector, prob, x_space);
		model = svm_train(&prob, &param);	
		cout<<svm_save_model(model_file_name, model)<<endl;
		free(prob.y);
		free(prob.x);
	}
	else{
		model = svm_load_model(model_file_name);
		free(prob.y);
		free(prob.x);
	}
	
	cout<<"model trained"<<endl;

	int classes = model->nr_class;
	double* probEstimate = new double[classes];
	probEstimate[0] = 1;
	probEstimate[1] = 1;

	int libsvm_correct = 0;
	int libsvm_correct_prob = 0;
	int libsvm_row = 0;

	struct svm_node *temp = NULL;
	for(int i = 0; i < test_rows * (test_cols + 1); i+=401)
	{
		temp = &test_x_space[i];
		libsvm_row = i / (test_cols + 1);

		int predicted = svm_predict(model, temp);
		int predicted_prob = svm_predict_probability(model, temp, probEstimate);
		//cout<<"temp is: "<<temp<<endl;
		//cout<<"testNode is: "<<temp->index<<" "<<temp->value<<endl;
		//cout<<"predicted value is: "<<predicted<<endl;
		//cout<<"Predicted value probability is: "<<predicted_prob<<endl;
		//cout<<"class probabilities are: "<<probEstimate[0]<<" "<<probEstimate[1]<<endl;	
		//cout<<endl;
		if(test_label_vector[libsvm_row] == predicted){
			libsvm_correct += 1;
		}
		if(test_label_vector[libsvm_row] == predicted_prob){
			libsvm_correct_prob += 1;
		}
	}

	cout<<"libsvm_correct is: "<<(double)libsvm_correct / (double)test_rows<<endl;
	cout<<"libsvm_correct_prob is: "<<(double)libsvm_correct_prob/(double)test_rows<<endl;



 //   // Set up SVM's parameters
 //   CvSVMParams params;
 //   params.svm_type    = CvSVM::C_SVC;
 //   params.kernel_type = CvSVM::RBF;
	//params.degree = 3;
	//params.gamma = 0.00001;
 //   params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

 //   // Train the SVM
 //   CvSVM SVM;
 //   SVM.train(train_data, label_data, Mat(), Mat(), params);

	//float correct = 0;
	//// Test the labels
	//for(int i = 0; i < test_length; i ++){
	//	Mat sampleMat = test_data.row(i);

	//	float response = SVM.predict(sampleMat);
	//	if(response == testLabel_data.at<float>(i,0)){
	//		correct++;
	//	}

	//}
	//float correct_ratio = correct / (float)test_length;
	//cout<<"The correct ratio is: "<<correct_ratio<<endl;
	
	int threshold_value = 50;
	int threshold_type = 3;
	int const max_value = 255;
	int const max_type = 4;
	int const max_BINARY_value = 255;
	//The particle segment image radius (half of width 20)
	int particle_image_radius = 10;
	//int particle_image_height = 10;
	Mat testImage = imread("C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\image processing\\test.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat dstImage;
	threshold(testImage, dstImage, threshold_value, max_BINARY_value, threshold_type);

	int image_height = dstImage.size().height;
	int image_width = dstImage.size().width;

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// detect edges using canny
	Canny( dstImage, canny_output, 50, 150, 3 );

	// find contours
	findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	// get the moments
	vector<Moments> mu(contours.size());
	for( int i = 0; i<contours.size(); i++ )
	{ mu[i] = moments( contours[i], false ); }
 
	// get the centroid of figures.
	vector<Point2f> mc(contours.size());

	//Padding the image so that particles are all in range
	int top, bottom, left, right;
	top = particle_image_radius;
	bottom = particle_image_radius;
	left = particle_image_radius;
	right = particle_image_radius;

	int borderType = 0;
	Scalar value = Scalar(0,0,0);

	Mat largeImage;
	copyMakeBorder( testImage, largeImage, top, bottom, left, right, borderType, value );
	// temporate variables for top left corner of particle segment
	float x, y = 0;
	// temporate particle segment image
	Mat particle_segment;
	Mat tempParticle;
	Size size(2,1);
	Mat detected_centroids = Mat::ones(1,2,CV_32FC1);//NOT good way to initiate a mat
	resize(detected_centroids, detected_centroids, size);

	Mat tempCentroid;
	float correct_contorus = 0;
	int predicted_particle;
	//temp svm_node containing the data of 
	struct svm_node* tempParticleNode = Malloc(svm_node,test_cols+1);
	for( int i = 0; i < contours.size(); i++){ 
		if(mu[i].m00 != 0){
			mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00);
			
			//cout<<"This is "<<i<<"th particle"<<endl;
	
			// get the particle segment image
			x = (int)mc[i].x;
			y = (int)mc[i].y;
			Rect region_of_interest = Rect(x, y, particle_image_radius * 2, particle_image_radius * 2);
			particle_segment = largeImage(region_of_interest);

		
			//namedWindow( "Display window", WINDOW_AUTOSIZE );
			//imshow( "Display window", particle_segment );  
			//waitKey(0);

			resize(particle_segment, tempParticle, Size(400, 1),0,0,1);//hard coded, modify later
			tempParticle.convertTo(tempParticle, CV_32FC1);
			//float result = SVM.predict(tempParticle, true);
			//float result0 = SVM.predict(tempParticle);

			//getSvmNomde<float>(test_cols, tempParticle, tempParticleNode);
			//predicted_particle = svm_predict_probability(model, tempParticleNode, probEstimate);

			pair<pair<int, int>, double> result = move_particle<int,double>(testImage, x, y, model, 1, 20, tempParticleNode);
			//cout<<"probability is: "<<result.second<<endl;
			//cout<<"The predicted value is: "<<predicted_particle<<endl;
			if(result.second >= 0.51/*probEstimate[1] >= 0.50*/){
				tempCentroid = Mat(mc[i]).clone().t();
				vconcat(detected_centroids,tempCentroid,detected_centroids);
			}

		}
	}

	free(tempParticleNode);
	// draw contours
	//Mat drawing(canny_output.size(), CV_8UC3, Scalar(255,255,255));
	//for( int i = 0; i<contours.size(); i++ )
	//{
	//Scalar color = Scalar(167,151,0); // B G R values
	//drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	//circle( drawing, mc[i], 10, color, 1, 8, 0 );
	//}
 
	//// show the resultant image
	//namedWindow( "Contours", WINDOW_AUTOSIZE );
	//imshow( "Contours", drawing );
	//waitKey(0);

	//Check the similiarity of the detected centroids and the real centroids
	cout<<"The siliarity between detected centroids and real centroids is: "<<check_similarity(test_centroids, detected_centroids, 1)<<endl;
	
	delete []probEstimate;
	probEstimate = NULL;
	
	return 0;
}

/*----------------------------
 * 功能 : 从 .txt 文件中读入数据，保存到 cv::Mat 矩阵
 *		- 默认按 float 格式读入数据，
 *		- 如果没有指定矩阵的行、列和通道数，则输出的矩阵是单通道、N 行 1 列的
 *----------------------------
 * 函数 : LoadData
 * 访问 : public 
 * 返回 : -1：打开文件失败；0：按设定的矩阵参数读取数据成功；1：按默认的矩阵参数读取数据
 *
 * 参数 : fileName	[in]	文件名
 * 参数 : matData	[out]	矩阵数据
 * 参数 : matRows	[in]	矩阵行数，默认为 0
 * 参数 : matCols	[in]	矩阵列数，默认为 0
 * 参数 : matChns	[in]	矩阵通道数，默认为 0
 */
template <class myType>
int LoadData(string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0)
{
	int retVal = 0;
 
	// 打开文件
	ifstream inFile(fileName.c_str(), ios_base::in);
	if(!inFile.is_open())
	{
		cout << "fail to read files" << endl;
		retVal = -1;
		return (retVal);
	}
 
	// 载入数据
	istream_iterator<myType> begin(inFile);	//按 float 格式取文件数据流的起始指针
	istream_iterator<myType> end;			//取文件流的终止位置
	vector<myType> inData(begin,end);		//将文件数据保存至 std::vector 中
	cv::Mat tmpMat = cv::Mat(inData);		//将数据由 std::vector 转换为 cv::Mat
 
	// 输出到命令行窗口
	//copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t")); 
 
	// 检查设定的矩阵尺寸和通道数
	size_t dataLength = inData.size();
	//1.通道数
	if (matChns == 0)
	{
		matChns = 1;
	}
	//2.行列数
	if (matRows != 0 && matCols == 0)
	{
		matCols = dataLength / matChns / matRows;
	} 
	else if (matCols != 0 && matRows == 0)
	{
		matRows = dataLength / matChns / matCols;
	}
	else if (matCols == 0 && matRows == 0)
	{
		matRows = dataLength / matChns;
		matCols = 1;
	}
	//3.数据总长度
	if (dataLength != (matRows * matCols * matChns))
	{
		cout << "input data length not match with the dimension and channel, going to input as default dimensions!" << endl;
		retVal = 1;
		matChns = 1;
		matRows = dataLength;
	} 
 
	// 将文件数据保存至输出矩阵
	matData = tmpMat.reshape(matChns, matRows).clone();
	
	return (retVal);
}
//--------------------- 
//作者：邹宇华 
//来源：CSDN 
//原文：https://blog.csdn.net/chenyusiyuan/article/details/7974378 
//版权声明：本文为博主原创文章，转载请附上博文链接！

// Function to return an absolute value
template <class myType>
myType absolute(myType val){
	if(val < 0)
		return -val;
	else
		return val;
}

// Function to delete a row in opencv Mat
Mat deleteRow(Mat a, int row){
	if(row <= a.rows){
		Mat b, c;
		a(Range(0, row), Range(0,a.cols)).copyTo(b);
		a(Range(row + 1, a.rows), Range(0,a.cols)).copyTo(c);

		vconcat(b,c,b);
		return b;	
	}
	else{
		cout<<"index out of matrix bound"<<endl;
		return a;
	}

}

// Funciton to check the similarity of two Mats of centroids
float check_similarity(Mat a, Mat b, float threshold){
	int aH = a.rows;
	int aW = a.cols;
	int bH = b.rows;
	int bW = b.cols;

	int ssize = 0;
	int lsize = 0;
	Mat small, large;

	if(aH > bH){
		ssize = bH;
		lsize = aH;
		b.copyTo(small);
		a.copyTo(large);
	}
	else{
		ssize = aH;
		lsize = bH;
		b.copyTo(large);
		a.copyTo(small);
	}

	float count = 0;
	float mag = 0;
	float dist = 0;
	vector<float> result(2);

	vector<float> s(2);
	vector<float> l(2); 

	for(int i = 0; i < ssize; i++){
		for(int j = 0;  j < lsize; j++){
			small.row(i).copyTo(s);
			large.row(j).copyTo(l);
			//cout<<"s is: "<<s[0]<<" "<<s[1]<<" l is: "<<l[0]<<" "<<l[1]<<endl;
			subtract(s, l, result);
			mag = result[0] * result[0] + result[1] * result[1];

			dist = sqrt(mag);
			//cout<<"mag is: "<<mag<<" dist is: "<<dist<<endl;
			if(dist <= threshold)
			{
				count += 1;
				break;
			}

		}
	}
	cout<<"count is: "<<count<<" lsize is: "<<lsize<<endl;
	return count / lsize;
}

// Function to calculate the svm possibility
// distance is the distance from boundary
float poss(float distance)
{
	return 1/(1+exp(-distance));
}

// Function to transform Mat row to vector<vector<>>
template <class myType>
vector<myType> matToVector_1d(Mat &m){
	int width = m.rows;

	const myType * p = m.ptr<myType>(0);
	vector<myType> a(p, p + width);
	cout<<"1d size is: "<<a.size()<<endl;
	return a;
}

// Function to transform 2d Mat to vector<vector<>>
template <class myType>
vector<vector<myType>> matToVector_2d(Mat &m){
	int height = m.rows;
	int width = m.cols;

	vector<vector<myType>> a;
	//Mat temp;
	for(int i = 0; i < height; i++){
		
		//m.row(i).copyTo(temp);
		
		const myType * p = m.ptr<myType>(i);
		vector<myType> v(p, p + width);

		
		a.push_back(v);
	}

	return a;
}

// Function to convert 2d vector to processed data for libsvm (Train data in libsvm model)
template <class dataType, class labelType>
void process_Train_Data(int sizeOfProblem, int elements, vector<vector<dataType>> &data, vector<labelType> &labels, svm_problem &prob, svm_node * &x_space){
		cout<<"data size is: "<<sizeOfProblem<<endl;

		cout<<"label size is: "<<elements<<endl;

		prob.l = sizeOfProblem;

		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(struct svm_node*, prob.l);

		x_space = Malloc(struct svm_node, (elements+1) * prob.l);

		// initialize the labels
		for(int i=0; i < prob.l; i++){
			prob.y[i] = labels[i];
		}
		cout<<"begin to transform data"<<endl;
		// initialze the svm_node vector with input data array
		int j=0; //counter to traverse x_space[i];
		for (int i=0;i < prob.l; ++i)
		{
			//set i-th element of prob.x to the address of x_space[j]. 
			//elements from x_space[j] to x_space[j+data[i].size] get filled right after next line
			prob.x[i] = &x_space[j];
			for (int k=0; k<data[i].size(); ++k, ++j)
			{
				x_space[j].index=k+1; //index of value
				x_space[j].value=data[i][k]; //value
				//cout<<"x_space["<<j<<"].index = "<<x_space[j].index;
				//cout<<"---x_space["<<j<<"].value = "<<x_space[j].value<<endl;
			}
			x_space[j].index=-1;//state the end of data vector
			x_space[j].value=0;
			//cout<<"x_space["<<j<<"].index = "<<x_space[j].index;
			//cout<<"---x_space["<<j<<"].value = "<<x_space[j].value<<endl;
			j++;
			//cout<<"j is: "<<j<<endl;
		}
		cout<<"data processed from vector to prob"<<endl;
}

// Function to convert 2d vector to processed data for libsvm (Test data)
template <class dataType>
void process_Test_Data(int sizeOfProblem, int elements, vector<vector<dataType>> &data, svm_node * &x_space){
		cout<<"data size is: "<<sizeOfProblem<<endl;

		cout<<"label size is: "<<elements<<endl;

		x_space = Malloc(struct svm_node, (elements+1) * sizeOfProblem);

		cout<<"begin to transform data"<<endl;
		// initialze the svm_node vector with input data array
		int j=0; //counter to traverse x_space[i];
		for (int i=0;i < sizeOfProblem; ++i)
		{
			for (int k=0; k<data[i].size(); ++k, ++j)
			{
				x_space[j].index=k+1; //index of value
				x_space[j].value=data[i][k]; //value
				//cout<<"x_space["<<j<<"].index = "<<x_space[j].index;
				//cout<<"---x_space["<<j<<"].value = "<<x_space[j].value<<endl;
			}
			x_space[j].index=-1;//state the end of data vector
			x_space[j].value=0;
			//cout<<"x_space["<<j<<"].index = "<<x_space[j].index;
			//cout<<"---x_space["<<j<<"].value = "<<x_space[j].value<<endl;
			j++;
			//cout<<"j is: "<<j<<endl;
		}
		cout<<"data processed from vector to prob"<<endl;
}

// Function to create svm_node
// Here m is supposed as a (N * 1) vector
template <class myType>
void getSvmNomde(int width, Mat &m, svm_node* node){
	int i;
	for(i = 0; i < width; i ++){
		node[i].index = i;
		node[i].value = m.at<myType>(i);
		//cout<<"i is: "<<i<<"value is: "<<node[i].value<<endl;
	}
	//cout<<"should be here before"<<endl;
	node[i].index = -1;
	//cout<<"should be here after"<<endl;
	node[i].value = 0;
}

// Funciton to move particle around and get the highest probability image coordinates
template <class coodType, class probType>
pair<pair<coodType, coodType>, probType> move_particle(Mat &m, int x, int y, svm_model *model, int radius, int part_width, svm_node *& node){
	//Padding the image so that particles are all in range

	int borderType = 0;
	Scalar value = Scalar(0,0,0);

	Mat largeImage;
	copyMakeBorder(m, largeImage, radius, radius, radius, radius, borderType, value);

	int image_height = largeImage.rows;
	int image_width = largeImage.cols;

	//cout<<"image height is: "<<image_height<<" width is: "<<image_width<<endl;

	int left, right, top, bottom;
	Mat segment;
	Mat tempParticle;
	Rect region_of_interest;
	Mat particle_segment;
	double* probEstimate = new double[2];
	int pred_result;
	
	pair<pair<coodType, coodType>, probType> ret;
	ret.first = make_pair<coodType, coodType>(0,0);
	ret.second = 0.0;

	for(int i = 0; i < 2 * radius + 1; i++){
		for(int j = 0; j < 2 * radius + 1; j++){
			left = x + i - (int)(part_width / 2);
			right = left + part_width;
			top = y + j - (int)(part_width / 2);
			bottom = top + part_width;

			if(left >=0 && top >= 0 && right < image_width && bottom < image_height){
				region_of_interest = Rect(left, top, part_width, part_width);//? right? or mix up x, y
				//cout<<left<<" "<<right<<" "<<top<<" "<<bottom<<endl;
				particle_segment = largeImage(region_of_interest);

				/*namedWindow( "Display window", WINDOW_AUTOSIZE);
				imshow( "Display window", particle_segment );  
				waitKey(0);*/


				resize(particle_segment, tempParticle, Size(400, 1),0,0,1);
				tempParticle.convertTo(tempParticle, CV_32FC1);

				getSvmNomde<float>(part_width * part_width, tempParticle, node);

				pred_result = svm_predict_probability(model, node, probEstimate);
				//cout<<"pred_result is: "<<pred_result<<endl;
				//cout<<"ProbEstimate is: "<<probEstimate[0]<<" "<<probEstimate[1]<<endl;

				if(pred_result == 1 && probEstimate[0] >= ret.second){
					ret.first = make_pair<coodType, coodType>(i + x - radius, j + y - radius);//here x is width, y is height
					ret.second = probEstimate[0];
				}
				//cout<<"result second is: "<<ret.second<<endl;
			}
		}
	}
	delete []probEstimate;
	return ret;
}