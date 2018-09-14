/************************************************************************/
// This class includes face pose estimation algorithms
// contain an algorithm for find pose of a face by K Nearest Neighbour(KNN).
// There is another method to detect accuracy of KNN on a Dataset.
// The code is implemented in September 2016 by NRDC.
// All rights are reserved.
/************************************************************************/

#include "FacePoseEstimation.h"

FacePoseEstimation::FacePoseEstimation(void)
{
     K = 14;
	//-----------------------------------
	// read Training Data matrix from file
    ifstream inputfile("XML/pose50__step0.csv");
	string current_line;
	// vector allows you to add data without knowing the exact size beforehand
	vector< vector<float> > all_data;
	// Start reading lines as long as there are lines in the file
    while(getline(inputfile, current_line)){
		// Now inside each line we need to separate the cols
		vector<float> values;
		stringstream temp(current_line);
		string single_value;
		while(getline(temp,single_value,',')){
			// convert the string element to a integer value
			values.push_back(atof(single_value.c_str()));
		}
		// add the row to the complete data vector
		all_data.push_back(values);
	}

    // Now add all the data into a Mat element
	poseEstimationTrain = Mat::zeros((float)all_data.size(), (float)all_data[0].size(), CV_32F);
	// Loop over vectors and add the data
	for(int rows = 0; rows < (int)all_data.size(); rows++){
		for(int cols= 0; cols< (int)all_data[0].size(); cols++){
			poseEstimationTrain.at<float>(rows,cols) = all_data[rows][cols];
		}
	}

    //==================================================

    train_sample_count = poseEstimationTrain.size().height;
    feature_count = poseEstimationTrain.size().width - 1;

    CvMat* trainData = cvCreateMat( train_sample_count, feature_count, CV_32FC1 );
    CvMat* trainClasses = cvCreateMat( train_sample_count, 1, CV_32FC1 );

    cv::Mat trainDataTemp( poseEstimationTrain, cv::Rect(0,0,feature_count,train_sample_count));
    cv::Mat trainClassesTemp( poseEstimationTrain, cv::Rect(feature_count,0,1, train_sample_count));

    trainData = &CvMat(trainDataTemp);
    trainClasses = &CvMat(trainClassesTemp);

    knn = new CvKNearest(trainData, trainClasses, 0, false, K );
}


FacePoseEstimation::~FacePoseEstimation(void)
{
    delete knn;
}

void FacePoseEstimation::facePoseEstimationKNN(FrameInfo & frameInfo)
{
    vector<vector<Point2f>> facesLandmarks = frameInfo.facesLandmarks;
    float response;
    //nearests = cvCreateMat( 1, K, CV_32FC1);

    // =============================================
    // ... Test Data
    CvMat* testDataExample = cvCreateMat( 1, feature_count, CV_32FC1);


    CvMat* dist = cvCreateMat(1, K, CV_32FC1);
    //const float** neighbours = new float*[K];
    CvMat* neighbourResponses = cvCreateMat(1, K, CV_32FC1);

    vector<int> facesPose;
    for(int i=0;i<facesLandmarks.size();i++)
    {
        vector<Point2f> tempFaceLandmark = facesLandmarks[i];
        Mat featureTest(1,68, CV_32FC1);
        //.....................................
        // .. convert 68 points landmark to => 68 feature
        float max = tempFaceLandmark[0].x;
        float min = tempFaceLandmark[0].x;
        // tempFaceLandmark[j], &max, &min

        for(int j=0;j<tempFaceLandmark.size();j++)
        {
            if(tempFaceLandmark[j].x > max)
                max = tempFaceLandmark[j].x;
            if(tempFaceLandmark[j].x < min)
                min = tempFaceLandmark[j].x;
        }
        float middle = (max- min)/2 + min;


        for(int j=0;j<tempFaceLandmark.size();j++)
            featureTest.at<float>(j) = abs(tempFaceLandmark[j].x - middle);
        featureTest = featureTest / norm(featureTest, NORM_L2);

        //.....................................

        testDataExample = &CvMat(featureTest);
        // estimate the response and get the neighbors' labels

        response = knn->find_nearest(testDataExample,K, 0, 0, neighbourResponses, dist);


        Mat distancePose = Mat(dist);
        distancePose = 1 / distancePose;
        distancePose = distancePose / sum(distancePose)[0];

        Mat pose;
        multiply(distancePose,Mat(neighbourResponses), pose);

        response = sum(pose)[0];

        facesPose.push_back(response);

    }

    frameInfo.facesPose = facesPose;
}

void FacePoseEstimation::facePoseEstimationKNN_TrainTest()
{
	float response;
	const int K = 14;

	int train_sample_count = poseEstimationTrain.size().height * 0.6;
	int test_sample_count = poseEstimationTrain.size().height - train_sample_count;
	int feature_count = poseEstimationTrain.size().width - 1;

	CvMat* trainData = cvCreateMat( train_sample_count, feature_count, CV_32FC1 );
	CvMat* testData = cvCreateMat( test_sample_count, feature_count, CV_32FC1 );
    CvMat* trainClasses = cvCreateMat( train_sample_count, 1, CV_32FC1 );
	CvMat* testClasses = cvCreateMat( test_sample_count, 1, CV_32FC1 );

	cv::Mat trainDataTemp( poseEstimationTrain, cv::Rect(0,0,feature_count,train_sample_count));
	cv::Mat testDataTemp( poseEstimationTrain, cv::Rect(0,train_sample_count, feature_count, test_sample_count));
	cv::Mat trainClassesTemp( poseEstimationTrain, cv::Rect(feature_count,0,1, train_sample_count));
	cv::Mat testClassesTemp( poseEstimationTrain, cv::Rect(feature_count, train_sample_count, 1, test_sample_count));

	trainData = &CvMat(trainDataTemp);
	testData = &CvMat(testDataTemp);
	trainClasses = &CvMat(trainClassesTemp);
	testClasses = &CvMat(testClassesTemp);

	CvKNearest knn( trainData, trainClasses, 0, false, K );
    CvMat* nearests = cvCreateMat( 1, K, CV_32FC1);

}

