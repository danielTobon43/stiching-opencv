#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/stitching.hpp"
#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>

#include "opencv2/core/core.hpp"

#include "eigen3/Eigen/Dense"
#include "opencv2/calib3d/calib3d.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

vector<Mat> imgs;
string result_name = "result.jpg";

void printUsage(char **argv) { std::cout << "Usage :\n" << argv[0] << " [images dir]" << std::endl; }

void keypointstoPoints2F(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Point2f> &points2D) {
  points2D.clear();
  for (const cv::KeyPoint &kp : keypoints) {
    points2D.push_back(kp.pt);
  }
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

void read_images_from_path(const std::string input_dir, std::vector<std::string> &images_dir,
                           std::vector<cv::Mat> &images_list_rgb);
void detect_and_extract_features(const std::vector<cv::Mat> &images_list_rgb, std::vector<cv::Mat> &images_list_grey,
                                 std::vector<std::vector<cv::KeyPoint>> &list_kps,
                                 std::vector<cv::Mat> &list_descriptors);
void compute_matching(const std::vector<cv::KeyPoint> &keypoints_image1,
                      const std::vector<cv::KeyPoint> &keypoints_image2, const cv::Mat &descriptor_image1,
                      const cv::Mat &descriptor_image2, std::vector<cv::DMatch> &good_matches);
void alignedPoints(const std::vector<cv::Point2f> &points_image1, const std::vector<cv::Point2f> &points_image2,
                   const std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &points_image1_aligned,
                   std::vector<cv::Point2f> &points_image2_aligned);
void get_ImageFeatures(const cv::Mat &image1, const cv::Mat &image2, std::vector<cv::KeyPoint> &keypoints_image1,
                       std::vector<cv::KeyPoint> &keypoints_image2, cv::Mat &descriptor_image1,
                       cv::Mat &descriptor_image2);

cv::Mat inverse(const cv::Mat &matrix) {

  Eigen::MatrixXd invMatrix, invMatrixTranspose;
  // Eigen::MatrixXd invMatrix;
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigenMatrix((double *)matrix.data,
                                                                                                 3, 3);
  std::cout << "eigen matrix: \n" << eigenMatrix << std::endl;

  invMatrix = eigenMatrix.inverse();
  std::cout << "eigen matrix inverse: \n" << invMatrix << std::endl;
  invMatrixTranspose = invMatrix.transpose();
  // std::cout << "eigen matrix inverse transpose: \n" << invMatrixTranspose << std::endl;
  // create an OpenCV Mat header for the Eigen data:
  // cv::Mat inv(invMatrix.rows(), invMatrix.cols(), CV_64FC1, invMatrix.data());
  cv::Mat inv(invMatrixTranspose.rows(), invMatrixTranspose.cols(), CV_64FC1, invMatrixTranspose.data());
  cv::Mat result;
  // inv.copyTo(inv_matrix);
  inv.copyTo(result);
  // inv_matrix(invMatrixTranspose.rows(), invMatrixTranspose.cols(), CV_64FC1, invMatrixTranspose.data());
  // std::cout << "cv MAT internal matrix: \n" << inv << std::endl;

  return result;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: <executable> <images path>" << std::endl;
    std::exit(-1);
  }

  std::cout << "\n*************************************" << std::endl;
  std::cout << "*** STITCHING IMAGES ***               " << std::endl;
  std::cout << "*************************************\n" << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// DATA ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  std::string input_dir = argv[1];
  std::vector<std::string> images_dir;
  std::vector<cv::Mat> images_list_rgb;
  std::vector<cv::Mat> images_list_grey;
  std::vector<std::vector<cv::KeyPoint>> list_kps;
  std::vector<cv::Mat> list_descriptors;
  std::vector<std::vector<cv::DMatch>> list_matches;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// READ IMAGES FROM DIRECTORY //////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  read_images_from_path(input_dir, images_dir, images_list_rgb);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// DETECT AND EXTRACT FEATURES  ////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  detect_and_extract_features(images_list_rgb, images_list_grey, list_kps, list_descriptors);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// COMPUTE MATCHING  ///////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  cv::Mat panorama;

  std::cout << "-> stitching images...please wait!" << std::endl;
  // cv::Mat current_homography = cv::Mat::ones(cv::Size(3, 3), CV_32FC1);
  // std::cout << current_homography << std::endl;

  cv::Mat previous_homography;
  cv::Mat current_homography;
  cv::Mat result;

  int cont = 0;

  for (int id_image1 = 0; id_image1 < images_list_rgb.size(); id_image1++) {

    for (int id_image2 = id_image1 + 1; id_image2 < images_list_rgb.size(); id_image2++) {

      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "id img1 :" << id_image1 << std::endl;
      std::cout << "id img2 :" << id_image2 << std::endl;

      cv::Mat image1;
      cv::Mat image2;

      // image1 = images_list_rgb[id_image1];
      // image2 = images_list_rgb[id_image2];

      image1.release();

      if (cont > 0) {

        // image1 = panorama;
        // panorama.copyTo(image1);
        // image1 = panorama.clone();
        image1 = result.clone();
        image2 = images_list_grey[id_image2].clone();
        std::cout << "type of homography:\n" << type2str(image1.type()) << std::endl;
        std::cout << "type of homography:\n" << type2str(image2.type()) << std::endl;
      } else {
        // image1 = panorama.clone();
        image1 = images_list_grey[id_image1].clone();
        image2 = images_list_grey[id_image2].clone();
        std::cout << "type of homography:\n" << type2str(image1.type()) << std::endl;
        std::cout << "type of homography:\n" << type2str(image2.type()) << std::endl;
        // image1 = panorama;
        // image2 = images_list_rgb[id_image2];
        // panorama.copyTo(image1);
        // images_list_rgb[id_image2].copyTo(image2);
      }

      std::vector<cv::KeyPoint> keypoints_image1 = list_kps[id_image1];
      std::vector<cv::KeyPoint> keypoints_image2 = list_kps[id_image2];
      cv::Mat descriptor_image1 = list_descriptors[id_image1];
      cv::Mat descriptor_image2 = list_descriptors[id_image2];

      std::vector<cv::DMatch> good_matches;

      compute_matching(keypoints_image1, keypoints_image2, descriptor_image1, descriptor_image2, good_matches);

      std::cout << "number of matches: " << good_matches.size() << std::endl;

      cv::Mat img_matches;
      drawMatches(image1, keypoints_image1, image2, keypoints_image2, good_matches, img_matches);
      cv::namedWindow("matches", cv::WINDOW_NORMAL);
      cv::resizeWindow("matches", 640, 480);
      cv::imshow("matches", img_matches);
      cv::waitKey(0);

      std::vector<cv::Point2f> points2d_image1;
      std::vector<cv::Point2f> points2d_image2;

      for (int i = 0; i < good_matches.size(); i++) {
        //--Get the keypoints from the good matches
        points2d_image1.push_back(keypoints_image1[good_matches[i].queryIdx].pt);
        points2d_image2.push_back(keypoints_image2[good_matches[i].trainIdx].pt);
      }

      previous_homography = current_homography.clone();

      if (cont > 0) {
        current_homography =
            (previous_homography)*inverse(cv::findHomography(points2d_image1, points2d_image2, cv::RANSAC));
      } else {
        current_homography = inverse(cv::findHomography(points2d_image1, points2d_image2, cv::RANSAC));
      }
      // cv::Mat offset_matrix = cv::Mat::eye(cv::Size(3, 3), CV_32F);
      // offset_matrix.at<float>(0, 2) = 0; // move forward x=600
      // offset_matrix.at<float>(1, 2) = 0; // move forward y

      // std::cout << "offset_matrix:\n" << offset_matrix << std::endl;

      // std::cout << "type of homography:\n" << type2str(current_homography.type()) << std::endl;
      // current_homography.convertTo(current_homography, CV_32F);
      // current_homography *= offset_matrix;

      // if (good_matches.size() >= 4) {

      //   if (current_homography.empty()) {
      //     std::cout << "yeah nene 1" << std::endl;
      //     current_homography = cv::findHomography(points2d_image1, points2d_image2, cv::RANSAC, 1.0);
      //   } else {
      //     std::cout << "yeah nene 2" << std::endl;
      //     current_homography *= cv::findHomography(points2d_image1, points2d_image2, cv::RANSAC);
      //     inverse(current_homography).copyTo(current_homography);
      //   }

      //   // homography = cv::getPerspectiveTransform(points2d_image1_aligned, points2d_image2_aligned);
      // } else {
      //   std::cout << "number of matches not good!" << std::endl;
      //   continue;
      // }

      // perspectiveTransform(points2d_image1, points2d_image2, current_homography);
      // int maxCols(0), maxRows(0);

      // for (int i = 0; i < points2d_image2.size(); i++) {
      //   if (maxRows < points2d_image2.at(i).y)
      //     maxRows = points2d_image2.at(i).y;
      //   if (maxCols < points2d_image2.at(i).x)
      //     maxCols = points2d_image2.at(i).x;
      // }

      std::cout << "homography matrix: \n" << current_homography << std::endl;

      // cv::Mat mask(image1.rows, image1.cols, CV_8UC1, Scalar(0));
      // cv::Mat mask = cv::Mat::zeros(result.size(), CV_8UC1);

      if (cont > 0) {
        // cv::Mat new_result;
        cv::warpPerspective(image2, result, current_homography, cv::Size(image2.cols * 2, image2.rows),
                            cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS, cv::BORDER_CONSTANT, cv::Scalar());

        cv::namedWindow("perspective", cv::WINDOW_NORMAL);
        cv::resizeWindow("perspective", 640, 480);
        cv::imshow("perspective", result);
        cv::waitKey(0);

        // new_result.copyTo(re);

        cv::Mat half2(result, cv::Rect(0, 0, image2.cols, image2.rows));

        cv::Mat croppedFrame = image1(cv::Rect(0, 0, image1.cols / 2, image1.rows));

        cv::namedWindow("crop", cv::WINDOW_NORMAL);
        cv::resizeWindow("crop", 640, 480);
        cv::imshow("crop", croppedFrame);
        cv::waitKey(0);

        // for (int i = 0; i < half2.rows; i++) {
        //   for (int j = 0; j < half2.cols; j++) {
        //     half2.at<cv::Vec3b>(i, j) = image1.at<cv::Vec3b>(i, j);
        //   }
        // }

        croppedFrame.copyTo(half2);
        // new_result.copyTo(result);

      } else {
        cv::warpPerspective(image2, result, current_homography, cv::Size(image2.cols * 2, image2.rows),
                            cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS, cv::BORDER_CONSTANT, cv::Scalar());

        cv::namedWindow("perspective", cv::WINDOW_NORMAL);
        cv::resizeWindow("perspective", 640, 480);
        cv::imshow("perspective", result);
        cv::waitKey(0);

        cv::Mat half(result, cv::Rect(0, 0, image1.cols, image1.rows));

        cv::namedWindow("half", cv::WINDOW_NORMAL);
        cv::resizeWindow("half", 640, 480);
        cv::imshow("half", half);
        cv::waitKey(0);

        image1.copyTo(half);
        std::string filename2 = "final_result_ppp.png";

        cv::imwrite(filename2, result);
      }

      cv::namedWindow("image1", cv::WINDOW_NORMAL);
      cv::resizeWindow("image1", 640, 480);
      cv::imshow("image1", image1);
      cv::waitKey(0);

      cv::namedWindow("image2", cv::WINDOW_NORMAL);
      cv::resizeWindow("image2", 640, 480);
      cv::imshow("image2", image2);
      cv::waitKey(0);

      cv::namedWindow("stitching", cv::WINDOW_NORMAL);
      cv::resizeWindow("stitching", 640, 480);
      cv::imshow("stitching", result);
      cv::waitKey(0);

      std::string filename = "final_result_" + std::to_string(cont) + ".png";
      cont += 1;
      cv::imwrite(filename, result);

      // panorama = result.clone();

      // result.copyTo(panorama);
      panorama = result.clone();

      // result.release();
      // half.release();

      // break;

      /* To remove the black portion after stitching, and confine in a rectangular region*/

      // std::vector<cv::Point> nonBlackList;
      // nonBlackList.reserve(result.rows * result.cols);

      // // add all non-black points to the vector
      // // there are more efficient ways to iterate through the image
      // for (int j = 0; j < result.rows; ++j)
      //   for (int i = 0; i < result.cols; ++i) {
      //     // if not black: add to the list
      //     if (result.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0)) {
      //       nonBlackList.push_back(cv::Point(i, j));
      //     }
      //   }

      // // create bounding rect around those points
      // cv::Rect bb = cv::boundingRect(nonBlackList);
      // result(bb).clone().copyTo(panorama);

      // half.release();
      // result.release();
      // result(bb).release();

      // display result and save it
      cv::destroyAllWindows();
      break;
    }
  }
  std::cout << "\nVisualizing results...\n=============================================\n";
  cv::destroyAllWindows();

  std::cout << "stitching completed successfully\n" << result_name << " saved!" << std::endl;
  return 0;
}

void read_images_from_path(const std::string input_dir, std::vector<std::string> &images_dir,
                           std::vector<cv::Mat> &images_list_rgb) {

  std::cout << "-> Getting images - " << std::flush;

  if (input_dir.size() > 0) {

    boost::filesystem::path dirPath(input_dir);

    if (not boost::filesystem::exists(dirPath) or not boost::filesystem::is_directory(dirPath)) {
      std::cout << "-> Error. cannot open directory: " << input_dir << std::endl;
      std::exit(-1);
    }

    bool foundImages;

    for (boost::filesystem::directory_entry &x : boost::filesystem::directory_iterator(dirPath)) {
      std::string extension = x.path().extension().string();
      boost::algorithm::to_lower(extension);
      if (extension == ".jpg" or extension == ".png") {
        foundImages = true;
        // std::cout << x.path().string() << std::endl;
        images_dir.push_back(x.path().string());
      }
    }

    if (not foundImages) {
      std::cout << "-> Unable to find images files in directory (\"" << input_dir << "\")." << std::endl;
      std::exit(-1);
    }
  } else {
    std::cout << "-> Unable to find images files in directory (\"" << input_dir << "\")." << std::endl;
    std::exit(-1);
  }

  std::cout << "[OK]" << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;
  std::sort(images_dir.begin(), images_dir.end());
  int pyramid_level = 2;
  for (std::vector<std::string>::iterator it = images_dir.begin(); it != images_dir.end(); ++it) {
    std::cout << *it << std::endl;
    cv::Mat image;
    image = cv::imread(cv::samples::findFile(*it), cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cout << "Could not open or find the image" << std::endl;
      std::exit(-1);
    } else {

      ////////////////////////////////////////////////////////////////////////////////////////////////
      ////////// DOWNSAMPLING IMAGE //////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////

      cv::Mat downsampled_image;
      if (image.rows > 480 and image.cols > 640) {
        cv::resize(image, downsampled_image, cv::Size(), 0.60, 0.60);
        images_list_rgb.push_back(downsampled_image);
      } else {
        images_list_rgb.push_back(downsampled_image);
      }

      // for (int i = 0; i < pyramid_level; i++) {
      //   cv::pyrDown(image, downsampled_image);
      // }
      // images_list_rgb.push_back(downsampled_image);
    }
  }
}

void detect_and_extract_features(const std::vector<cv::Mat> &images_list_rgb, std::vector<cv::Mat> &images_list_grey,
                                 std::vector<std::vector<cv::KeyPoint>> &list_kps,
                                 std::vector<cv::Mat> &list_descriptors) {

  list_kps.clear();
  list_descriptors.clear();
  images_list_grey.clear();
  images_list_grey.resize(images_list_rgb.size());
  list_kps.resize(images_list_rgb.size());
  list_descriptors.resize(images_list_rgb.size());

  std::cout << "Detecting features...please wait" << std::endl;

  for (std::vector<cv::Mat>::const_iterator it = images_list_rgb.begin(); it != images_list_rgb.end(); ++it) {

    cv::Mat image = *it;
    int idx = std::distance(images_list_rgb.begin(), it);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////// CONVERTION TO GREYSCALE /////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    cv::Mat image_grey;
    cv::cvtColor(image, image_grey, cv::COLOR_RGB2GRAY);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////// FILTERING NOISE /////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // cv::Mat image_blur;
    // cv::GaussianBlur(image_grey, image_blur, cv::Size(3, 3), 0.75, 0.75);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////// DETECTING FEATURES //////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;

    //----------------- AKAZE -------------------------
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    // akaze->detectAndCompute(image_grey, cv::noArray(), keypoints, descriptor);

    //----------------- ORB -------------------------
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // orb->detectAndCompute(image_grey, cv::noArray(), keypoints, descriptor);

    //----------------- SIFT -------------------------
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    // sift->detectAndCompute(image_grey, cv::noArray(), keypoints, descriptor);

    //----------------- SURF -------------------------
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    surf->detectAndCompute(image_grey, cv::noArray(), keypoints, descriptor);

    // cv::Mat outImage;
    // cv::drawKeypoints(image_grey, keypoints, outImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::namedWindow("features", cv::WINDOW_NORMAL);
    // cv::resizeWindow("features", 640, 480);
    // imshow("features", outImage);
    // cv::waitKey(400);

    // list_descriptors[i] = descriptor;
    images_list_grey[idx] = image_grey;
    list_kps[idx] = keypoints;
    list_descriptors[idx] = descriptor;
  }
}

void get_ImageFeatures(const cv::Mat &image1, const cv::Mat &image2, std::vector<cv::KeyPoint> &keypoints_image1,
                       std::vector<cv::KeyPoint> &keypoints_image2, cv::Mat &descriptor_image1,
                       cv::Mat &descriptor_image2) {

  keypoints_image1.clear();
  keypoints_image2.clear();

  std::cout << "Detecting features...please wait" << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// CONVERTION TO GREYSCALE /////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  cv::Mat image1_grey;
  cv::Mat image2_grey;
  cv::cvtColor(image1, image1_grey, cv::COLOR_RGB2GRAY);
  cv::cvtColor(image2, image2_grey, cv::COLOR_RGB2GRAY);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// FILTERING NOISE /////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  // cv::Mat image1_blur;
  // cv::Mat image2_blur;
  // cv::GaussianBlur(image1_grey, image1_blur, cv::Size(3, 3), 0, 0);
  // v::GaussianBlur(image2_grey, image2_blur, cv::Size(3, 3), 0, 0);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////// DETECTING FEATURES //////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  //----------------- AKAZE -------------------------
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
  // akaze->detectAndCompute(image_grey, cv::noArray(), keypoints, descriptor);

  //----------------- ORB -------------------------
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  // orb->detectAndCompute(image1_grey, cv::noArray(), keypoints, descriptor);

  //----------------- SIFT -------------------------
  cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
  sift->detectAndCompute(image1_grey, cv::noArray(), keypoints_image1, descriptor_image1);
  sift->detectAndCompute(image2_grey, cv::noArray(), keypoints_image2, descriptor_image2);

  //----------------- SURF -------------------------
  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
  // surf->detectAndCompute(image_blur, cv::noArray(), keypoints, descriptor);
}

void compute_matching(const std::vector<cv::KeyPoint> &keypoints_image1,
                      const std::vector<cv::KeyPoint> &keypoints_image2, const cv::Mat &descriptor_image1,
                      const cv::Mat &descriptor_image2, std::vector<cv::DMatch> &good_matches) {

  // METHOD 1

  // const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
  // const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
  // good_matches.clear();

  // cv::BFMatcher matcher(cv::NORM_HAMMING);
  // std::vector<std::vector<DMatch>> nn_matches;
  // matcher.knnMatch(descriptor_image1, descriptor_image2, nn_matches, 2);
  // std::vector<cv::KeyPoint> matched1, matched2;

  // for (size_t i = 0; i < nn_matches.size(); i++) {
  //   cv::DMatch first = nn_matches[i][0];
  //   float dist1 = nn_matches[i][0].distance;
  //   float dist2 = nn_matches[i][1].distance;
  //   if (dist1 < nn_match_ratio * dist2) {
  //     matched1.push_back(keypoints_image1[first.queryIdx]);
  //     matched2.push_back(keypoints_image2[first.trainIdx]);
  //   }
  // }

  // std::vector<cv::Point2f> points2d_image1;
  // std::vector<cv::Point2f> points2d_image2;

  // // keypointstoPoints2F(matched1, points2d_image1);
  // points2d_image1.clear();
  // points2d_image2.clear();

  // points2d_image1.resize(matched1.size());
  // points2d_image2.resize(matched2.size());

  // for (int i = 0; i < matched1.size(); i++) {
  //   points2d_image1[i] = matched1[i].pt;
  // }

  // for (int i = 0; i < matched2.size(); i++) {
  //   points2d_image2[i] = matched2[i].pt;
  // }

  // // keypointstoPoints2F(matched2, points2d_image2);

  // // for (int i = 0; i < nn_matches[0].size(); i++) {
  // //   //--Get the keypoints from the good matches
  // //   points2d_image1.push_back(matched1[good_matches[i].queryIdx].pt);
  // //   points2d_image2.push_back(matched2[good_matches[i].trainIdx].pt);
  // // }

  // cv::Mat homography = cv::findHomography(points2d_image1, points2d_image2, cv::RANSAC);

  // // cv::Mat homography = cv::findHomography(matched1.pt, matched2.pt, cv::RANSAC);

  // std::vector<cv::KeyPoint> inliers1, inliers2;
  // for (size_t i = 0; i < matched1.size(); i++) {
  //   cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
  //   col.at<double>(0) = matched1[i].pt.x;
  //   col.at<double>(1) = matched1[i].pt.y;
  //   col = homography * col;
  //   col /= col.at<double>(2);
  //   double dist = std::sqrt(std::pow(col.at<double>(0) - matched2[i].pt.x, 2) +
  //                           std::pow(col.at<double>(1) - matched2[i].pt.y, 2));
  //   if (dist < inlier_threshold) {
  //     int new_i = static_cast<int>(inliers1.size());
  //     inliers1.push_back(matched1[i]);
  //     inliers2.push_back(matched2[i]);
  //     good_matches.push_back(cv::DMatch(new_i, new_i, 0));
  //   }
  // }

  // METHOD 2

  // /*Knn matching*/
  cv::BFMatcher *matcher = new cv::BFMatcher(cv::NORM_L2, false);
  // std::vector<cv::KeyPoint> query_kps = imagesKeypoints.at(idx_query);
  // std::vector<cv::KeyPoint> train_kps = imagesKeypoints.at(idx_train);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  cv::Mat query_descriptor = descriptor_image1;
  cv::Mat train_descriptor = descriptor_image2;
  matcher->knnMatch(query_descriptor, train_descriptor, knnMatches, 2);

  std::vector<cv::DMatch> matches_;

  /*RATIO-TEST FILTER*/
  float NN_MATCH_RATIO = 0.8f;

  for (unsigned i = 0; i < knnMatches.size(); i++) {
    if (knnMatches[i][0].distance <= NN_MATCH_RATIO * knnMatches[i][1].distance) {
      matches_.push_back(knnMatches[i][0]);
    }
  }

  const double ransac_thresh = 2.5;
  cv::Mat inliers_mask, H;

  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;

  for (int i = 0; i < matches_.size(); i++) {
    //--Get the keypoints from the good matches
    points1.push_back(keypoints_image1[matches_[i].queryIdx].pt);
    points2.push_back(keypoints_image2[matches_[i].trainIdx].pt);
  }

  if (matches_.size() >= 4) {
    H = cv::findHomography(points1, points2, cv::RANSAC, ransac_thresh, inliers_mask);
  }

  std::cout << "Homography inliers mask:" << inliers_mask.rows << " inliers" << std::endl;

  for (unsigned i = 0; i < inliers_mask.rows; i++) {
    if (inliers_mask.at<uchar>(i)) {
      // int new_i = static_cast<int>(inliers1.size());
      // inliers1.push_back(matched1[i]);
      // inliers2.push_back(matched2[i]);
      // goodMatches->push_back(cv::DMatch(new_i, new_i, 0));
      good_matches.push_back(matches_[i]);
    }
  }

  // METHOD 3
  // cv::Mat query_descriptor = descriptor_image1;
  // cv::Mat train_descriptor = descriptor_image2;

  // if (query_descriptor.type() != CV_32F) {
  //   query_descriptor.convertTo(query_descriptor, CV_32F);
  // }

  // if (train_descriptor.type() != CV_32F) {
  //   train_descriptor.convertTo(train_descriptor, CV_32F);
  // }

  // //-- Step 3: Matching descriptor vectors using FLANN matcher
  // cv::FlannBasedMatcher matcher;
  // std::vector<cv::DMatch> matches;
  // matcher.match(query_descriptor, train_descriptor, matches);

  // double max_dist = 0;
  // double min_dist = 100;

  // //-- Quick calculation of max and min distances between keypoints
  // for (int i = 0; i < query_descriptor.rows; i++) {
  //   double dist = matches[i].distance;
  //   if (dist < min_dist)
  //     min_dist = dist;
  //   if (dist > max_dist)
  //     max_dist = dist;
  // }

  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  // //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )

  // for (int i = 0; i < descriptor_image1.rows; i++) {
  //   if (matches[i].distance < 3 * min_dist) {
  //     good_matches.push_back(matches[i]);
  //   }
  // }

  // double inlier_ratio = inliers1.size() / (double)matched1.size();
  // std::cout << "A-KAZE Matching Results" << std::endl;
  // std::cout << "*******************************" << std::endl;
  // std::cout << "# Keypoints 1:                        \t" << keypoints_image1.size() << std::endl;
  // std::cout << "# Keypoints 2:                        \t" << keypoints_image2.size() << std::endl;
  // std::cout << "# Matches:                            \t" << matched1.size() << std::endl;
  // std::cout << "# Inliers:                            \t" << inliers1.size() << std::endl;
  // std::cout << "# Inliers Ratio:                      \t" << inlier_ratio << std::endl;
  // std::cout << std::endl;
}

void alignedPoints(const std::vector<cv::Point2f> &points_image1, const std::vector<cv::Point2f> &points_image2,
                   const std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &points_image1_aligned,
                   std::vector<cv::Point2f> &points_image2_aligned) {

  // align left and right point sets
  for (unsigned int i = 0; i < matches.size(); i++) {

    points_image1_aligned.push_back(points_image1[matches[i].queryIdx]);
    points_image2_aligned.push_back(points_image2[matches[i].trainIdx]);
  }
}