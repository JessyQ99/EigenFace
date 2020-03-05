
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include<fstream>
#include<sstream>
#define TRAIN_IMG_PATH "facedatabase/trainimg/"
#define TEST_IMG_PATH "facedatabase/testimg/"
#define TRAIN_IMG_LIST "facedatabase/trainlist.txt"
#define TEST_IMG_LIST "facedatabase/testlist.txt"
#define TRAIN_IMG_EYE_POSE "facedatabase/traindata.txt"
#define TEST_IMG_EYE_POSE "facedatabase/testdata.txt"
using namespace std;
using namespace cv;




class FaceAlignment //人脸图像预处理操作
{
public:
	Mat origin_pic;
	Mat gray_pic;
	Mat transformed_pic;
	double x1, y1, x2, y2;
	Mat trans_mat;
	Mat equalized_mat;
	Mat vect;
	int label;

	void load(string imgpath)
	{
		gray_pic = imread(imgpath, IMREAD_GRAYSCALE);
		align();
		equalizeHist(gray_pic, gray_pic); //进行直方图均衡化，使程序对光照不敏感
		resize(gray_pic, gray_pic, Size(240, 320));//进行图片统一大小操作,这里resize是因为原图计算量太大。
		vect = gray_pic.reshape(0, 1).t(); //将图片化为向量
	}

	void align() //人脸对齐函数，主要通过计算两眼位置和倾斜度来校正人脸位置
	{
		Point center((x1 + x2) / 2, (y1 + y2) / 2);
		double angle = atan((double)(y2 - y1) / (double)(x2 - x1)) * 180.0 / CV_PI; 
		trans_mat = getRotationMatrix2D(center, angle, 1.0); //使两眼处于水平状态
		trans_mat.at<double>(0, 2) += 285 - center.x; //平移图像使每张图像眼睛位置对齐
		trans_mat.at<double>(1, 2) += 349 - center.y;
		warpAffine(gray_pic, transformed_pic, trans_mat, gray_pic.size()); //进行几何变换操作


	}

};

class FaceLib //构建人脸识别库
{
public:
	int num_of_faces; //脸的总数
	int num_of_persons; //总共有几个对象
	int facepernum;//每人几张图像
	vector<FaceAlignment*> faces;
	vector<Mat_<double>> _samples;
	Mat_<double> samples;
	Mat_<double> graypic;
	vector<vector<double>>eye_pos;
	vector<string>filelist;
	string data_path;
	double x1, y1, x2, y2;
	void init(string eye_pose_path, int faces, int persons, int pernum)
	{

		num_of_faces = faces;
		num_of_persons = persons;
		facepernum = pernum;
		load_eye_pos(eye_pose_path, num_of_faces);
	}
	void load_eye_pos(string eye_pose_path, int num) //获取每张图像的眼睛位置数据，存储格式为左眼x,左眼y，右眼x，右眼y
	{
		ifstream in(eye_pose_path);
		for (int i = 0; i <num; i++)
		{
			vector<double>linedata;
			in >> x1 >> y1 >> x2 >> y2;
			linedata.push_back(x1);
			linedata.push_back(y1);
			linedata.push_back(x2);
			linedata.push_back(y2);
			eye_pos.push_back(linedata);


		}


	}
	void loadimg(string imglistpath, string rootPath) //加载图片
	{
		ifstream f(imglistpath); //先加载图片文件列表
		string filename;
		for (int i = 0; i < num_of_persons; i++)
		{
			for (int j = 0; j < facepernum; ++j)
			{
				f >> filename;
				filename = rootPath + filename + ".jpg"; //根据列表读取每张图片
				filelist.push_back(filename);
				FaceAlignment* face = new FaceAlignment();
				face->x1 = eye_pos[i*facepernum + j][0];
				face->y1 = eye_pos[i*facepernum + j][1];
				face->x2 = eye_pos[i*facepernum + j][2];
				face->y2 = eye_pos[i*facepernum + j][3];
				face->load(filename);
				face->label = i;
				faces.push_back(face);
				_samples.push_back(face->vect);

			}
		}
		hconcat(_samples, samples);
	}
};