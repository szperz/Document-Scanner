#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat img, imgCanny, imgGray, imgBlur, imgWarp;
std::vector<cv::Point> initialPoints, docPoints;
float w = 420, h = 596;

cv::Mat preProcess(cv::Mat img) {
	cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, cv::Size(5, 5), 5, 0);
	Canny(imgBlur, imgCanny, 30, 100);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	dilate(imgCanny, imgCanny, kernel);
	return imgCanny;
}

std::vector<cv::Point> getContours(cv::Mat imgCanny, cv::Mat img) {
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	findContours(imgCanny, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	
	std::vector<std::vector<cv::Point>> conPoly(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Point> biggest;
	int maxArea{ 0 };

	for(int i {0}; i < contours.size(); ++i) {
		int area = contourArea(contours[i]);
		if (area > 10000) {
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			if (area > maxArea && conPoly[i].size() == 4) {
				biggest = { conPoly[i][0],conPoly[i][1], conPoly[i][2], conPoly[i][3] };
				area = maxArea;
			}
		}
	}
	return biggest;
}

std::vector<cv::Point> reorder(std::vector<cv::Point> points) {
	std::vector<cv::Point> newPoints;
	std::vector<int> sumPoints, subPoints;

	for (int i{ 0 }; i < 4; ++i) {
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);

	return newPoints;
}

void drawPoints(std::vector<cv::Point> points, cv::Scalar color) {
	for (int i{ 0 }; i < points.size(); ++i) {
		circle(img, points[i], 4, color, cv::FILLED);
	}
}

cv::Mat Warp(cv::Mat img, std::vector<cv::Point> points, float w, float h)
{
	cv::Point2f src[4] = { points[0],points[1],points[2],points[3] };
	cv::Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	cv::Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, cv::Point(w, h));

	return imgWarp;
}

int main() {
	int cameraID{ 0 };
	std::cout << "Enter camera ID" << std::endl;
	std::cin >> cameraID;

	cv::VideoCapture cap(cameraID);

	while (true) {
		cap.read(img);
		imgCanny = preProcess(img); 
		initialPoints = getContours(imgCanny, img);  

		if (!initialPoints.empty()) {
			docPoints = reorder(initialPoints);
			imgWarp = Warp(img, docPoints, w, h);
			imshow("Your Document", imgWarp);
			imwrite("Resources/document.png", imgWarp);
			drawPoints(docPoints, cv::Scalar(0, 255, 255));
		}
		imshow("Image", img);
		cv::waitKey(1);
	}
	return 0;
}
