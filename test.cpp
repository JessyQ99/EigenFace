
#include"preprocess.h"
Mat Vec2Img(Mat vect, int width, int height)
{

	Mat result(Size(width, height), CV_64FC1);
	int i;
	for (i = 0; i < height; ++i)
	{
		vect.colRange(i*width, (i + 1)*width).convertTo(result.row(i), CV_64FC1);
	}
	normalize(result, result, 255, 0.0, NORM_MINMAX);
	result.convertTo(result,CV_8UC1);
	return result;

}

int main(int argc,char**argv)
{

	FaceLib*facelib=new FaceLib(); //加载测试集
	facelib->init(TEST_IMG_EYE_POSE, 123, 41, 3);
	facelib->loadimg(string(TEST_IMG_LIST),string(TEST_IMG_PATH));
	ofstream f("record111.txt"); //用来记录每张图片的预测值
	string model_name = argv[2];
	string model_name1 = argv[3];

	Mat e_vector_mat;
	Mat mean_mat;
	FileStorage model(model_name , FileStorage::READ); //读取提取的主要特征值
	FileStorage model1(model_name1, FileStorage::READ); //读取均值图像
	model["eigenvector"] >> e_vector_mat;
	model1["mean_mat"] >> mean_mat;
	Mat distance;
	Mat sample;
	FaceAlignment face;
	int idx = 0;
	
	FaceLib*src = new FaceLib(); //加载训练集
	src->init(TRAIN_IMG_EYE_POSE, 246, 41, 6);
	src->loadimg(string(TRAIN_IMG_LIST), string(TRAIN_IMG_PATH));
	int correctPre = 0;
	f << "person    " << "predicted" << endl;
	
	for (idx = 0; idx <123; idx ++)
	{
		Mat curValue = facelib->samples(Rect(idx, 0, 1, src->samples.rows));
		Mat curV1 = e_vector_mat * (curValue - mean_mat); //将测试集当前图像减去人脸均值
		Mat libValue = src->samples(Rect(0, 0, 1, src->samples.rows));
		Mat curV2 = e_vector_mat * (libValue - mean_mat); //将训练集当前图像减去人脸均值，进行降维
		string fpath = facelib->filelist[idx];
		Mat testMat = imread(fpath);
		double min_d = norm(curV1, curV2, NORM_L2); //计算两张图像间的距离，采用L2范式
		int min_idx = 0;
		int min_pic = 0;
		double dis;
		for (int pic = 0; pic < 246; pic++)
		{
			libValue = src->samples(Rect(pic, 0, 1, src->samples.rows));
			curV2 = e_vector_mat * (libValue - mean_mat);
			dis = norm(curV1, curV2, NORM_L2);
			if (dis < min_d) //更新最相似图像
			{
				min_d = dis;
				min_idx = pic / 6; //获取该图像属于哪个人
				min_pic = pic;  //更新最相似图片下标
			}
			
		}

		Mat most_smiliar=  src->samples(Rect(min_pic, 0, 1, src->samples.rows)).t();
		Mat re = Vec2Img(most_smiliar,240,320);
		imshow("Most similar:", re);//展现最相似的原图
		
		Mat re1 = Vec2Img(curValue.t(), 240, 320);
		imshow("current:", re1);

		Mat dst;
		addWeighted(re, 0.5, re1, 0.5, 0, dst);
		imshow("MergePic:", dst); //展现融合后的图像

		string text = "Person" + to_string(min_idx);
		putText(testMat, text, Point(10, 580), FONT_HERSHEY_COMPLEX, 0.5, (255,0,255)); //原图片（彩色图片）打上识别的人物结果标记
		string text2="Most similar:No." + to_string(min_pic);
		putText(testMat, text2, Point(10,610 ), FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255));//图片打上每张图片最相似的图片序号
		imwrite(fpath,testMat);
		int person = idx / 3;
		if (person == min_idx) //真实结果和识别结果一致
			correctPre += 1;
		f << person <<"    "<<min_idx<< endl;
		waitKey(0);
	}
	
	double ratio = 1.0*correctPre / 123; //计算正确率
	f<<"ratio:   "<<ratio<<endl;

	
}