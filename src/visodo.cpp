/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>


using namespace cv;
using namespace std;

void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1, vector<Point2f> &points2, vector<uchar> &status)
{

  // this function automatically gets rid of points for which tracking fails

  vector<float> err;
  Size winSize = Size(21, 21);
  TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  // getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for (int i = 0; i < status.size(); i++)
  {
    Point2f pt = points2.at(i - indexCorrection);
    if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
    {
      if ((pt.x < 0) || (pt.y < 0))
      {
        status.at(i) = 0;
      }
      points1.erase(points1.begin() + (i - indexCorrection));
      points2.erase(points2.begin() + (i - indexCorrection));
      indexCorrection++;
    }
  }
}

void featureDetection(Mat img_1, vector<Point2f> &points1)
{ // uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}


using namespace cv;
using namespace std;

int MAX_FRAME = 1200;
#define MIN_NUM_FEAT 2500

int main(int argc, char **argv)
{
  for (size_t k = 7; k < 8; k++)
  {
    std::string directory = std::to_string(k);

    if (k == 4)
      MAX_FRAME = 1196;

    Mat img_1, img_2;
    Mat R_f, t_f;

    // Mat Rx = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(M_PI / 2), -sin(M_PI / 2), 0, sin(M_PI / 2), cos(M_PI / 2));
    // Mat Ry = (Mat_<double>(3, 3) << cos(M_PI / 2), 0, sin(M_PI / 2), 0, 1, 0, -sin(M_PI / 2), 0, cos(M_PI / 2));
    // Mat Rz = (Mat_<double>(3, 3) << cos(M_PI / 2), -sin(M_PI / 2), 0, sin(M_PI / 2), cos(M_PI / 2), 0, 0, 0, 1);

    double scale = 1.00;
    char filename1[200];
    char filename2[200];
    sprintf(filename1, "/home/ernie/mono-vo/calib_challenge/unlabeled/%s/output_%03d.jpg", directory.c_str(), 1);
    sprintf(filename2, "/home/ernie/mono-vo/calib_challenge/unlabeled/%s/output_%03d.jpg", directory.c_str(), 2);

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data)
    {
      std::cout << " --(!) Error reading images " << std::endl;
      return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2; // vectors to store the coordinates of the feature points
    featureDetection(img_1, points1); // detect features in img_1
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status); // track those features to img_2

    double focal = 910.0;
    cv::Point2d pp(582.0, 437.0);
    // recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    double angle1 = 0;
    double angle2 = 0;
    int count = 0;

    std::ofstream log_file;
    log_file.open("log.txt");

    for (int numFrame = 3; numFrame <= MAX_FRAME; numFrame++)
    {
      if (k == 5 && numFrame == 1119) 
        break;
      sprintf(filename, "/home/ernie/mono-vo/calib_challenge/unlabeled/%s/output_%03d.jpg", directory.c_str(), numFrame);
      // cout << numFrame << endl;
      Mat currImage_c = imread(filename);
      cout << filename << endl;
      if (currImage_c.empty())
        continue;
      cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
      vector<uchar> status;
      featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

      E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
      recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

      Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

      for (int i = 0; i < prevFeatures.size(); i++)
      { // this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
        prevPts.at<double>(0, i) = prevFeatures.at(i).x;
        prevPts.at<double>(1, i) = prevFeatures.at(i).y;

        currPts.at<double>(0, i) = currFeatures.at(i).x;
        currPts.at<double>(1, i) = currFeatures.at(i).y;
      }

      scale = 1.0;
      if (t.at<double>(2) > .9 && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
      {
        t_f = t_f + scale * (R_f * t);
        R_f = R * R_f;

        // t = t / cv::norm(t);
        angle1 += std::atan2(t.at<double>(1), t.at<double>(2));
        angle2 += std::atan2(t.at<double>(0), t.at<double>(2));
        count++;

        log_file << std::atan2(t.at<double>(1), t.at<double>(2)) << " " << std::atan2(t.at<double>(0), t.at<double>(2)) << '\n';
      }

      // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
      if (prevFeatures.size() < MIN_NUM_FEAT)
      {
        featureDetection(prevImage, prevFeatures);
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
      }

      prevImage = currImage.clone();
      prevFeatures = currFeatures;

      // sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
      sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t.at<double>(0), t.at<double>(1), t.at<double>(2));
      // std::cout << text << std::endl;
    }

    cout << -angle1 / count << endl;
    cout << angle2 / count << endl;

    char outfile[100];
    sprintf(outfile, "../calib_challenge/unlabeled/%s_res.txt", directory.c_str());

    std::ofstream output_file;
    output_file.open(outfile);
    for (size_t i = 0; i < MAX_FRAME; i++)
      output_file << -angle1 / count << " " << angle2 / count << "\n";
    output_file.close();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;
  }

  return 0;
}

// https://docs.opencv.org/4.x/d0/d61/group__cudacodec.html#gga71943a1181287609b5d649f53ce6c146a201f327572b8d5df724115b075d2ffc0
// https://github.com/commaai/calib_challenge/tree/main/unlabeled