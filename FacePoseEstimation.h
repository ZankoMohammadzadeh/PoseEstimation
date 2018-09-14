#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <fstream>
#include "FrameInfo.h"

class FacePoseEstimation
{
private:
    int train_sample_count;
    int feature_count;
    int K;
    CvKNearest *knn;

	Mat poseEstimationTrain;
public:
	FacePoseEstimation(void);
	~FacePoseEstimation(void);

	void facePoseEstimationKNN_TrainTest();
	void facePoseEstimationKNN(FrameInfo &);

};

