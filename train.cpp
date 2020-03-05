
#include "preprocess.h"
Mat Vec2Img(Mat vect, int width, int height); //���ڽ�������ԭΪͼƬ


int main(int argc,char**argv) 
{
	
	double energy = atof(argv[1]);
	string model_name = argv[2];
	string model_name1 = argv[3];
	
	
	FaceLib*facelib=new FaceLib();
	facelib->init(string(TRAIN_IMG_EYE_POSE), 240, 40, 6);//����ѵ����
	facelib->loadimg(string(TRAIN_IMG_LIST),string(TRAIN_IMG_PATH));
	Mat samples, cov_mat, mean_mat,eigenValues,eigenVector;
	facelib->samples.copyTo(samples);
	FileStorage model(model_name, FileStorage::WRITE); //д��ǰK����������
	FileStorage model1(model_name1, FileStorage::WRITE);//д��������ֵ����
	calcCovarMatrix(samples, cov_mat, mean_mat, COVAR_COLS | COVAR_NORMAL); //����Э�������
	cov_mat = cov_mat / (samples.rows - 1); 
	eigen(cov_mat, eigenValues, eigenVector);
	vector<Mat> Top10; //����ǰ10����������ֵ����
	for (int i = 0; i < 10; i++)
	{
		Top10.push_back(Vec2Img(eigenVector.row(i), 240, 320));
	}
	Mat result;
	hconcat(Top10, result);//��10����������һ��

	result.convertTo(result, CV_8U, 255); //imshow��֧��CV_64FC1����ת��

	imshow("Top10EigenFace", result);
	imwrite("Top10EigenFace.png", result); //��ǰ10ͼ��д���ļ�

	double totalValue = sum(eigenValues)[0];
	double upEnergy = totalValue * energy;
	double energySum = 0;
	int k = 0;
	for (k = 0; k < eigenValues.rows; k++) //������Ҫ���������ֵ��Ŀ
	{
		energySum += eigenValues.at<double>(k, 0);
		if (energySum >= upEnergy) break;
	}

	eigenVector = eigenVector.rowRange(0, k); //����ǰK����������
	model << "eigenvector" << eigenVector;
	model1 << "mean_mat" << mean_mat;
	model.release();
	model1.release();


	return 0;
}

Mat Vec2Img(Mat vect, int width, int height)
{

	Mat result(Size(width, height), CV_64FC1);
	int i;
	for (i = 0; i < height; ++i)
	{
		vect.colRange(i*width, (i + 1)*width).convertTo(result.row(i), CV_64FC1);
	}
	normalize(result, result, 255, 0.0, NORM_MINMAX); //����һ������
	result.convertTo(result, CV_8UC1);
	return result;

}
