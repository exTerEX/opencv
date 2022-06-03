#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <tesseract/baseapi.h>

bool compare_contours(std::vector<cv::Point> &contour1,
                      std::vector<cv::Point> &contour2)
{
    const double contour_1 = std::fabs(cv::contourArea(cv::Mat(contour1)));
    const double contour_2 = std::fabs(cv::contourArea(cv::Mat(contour2)));

    return (contour_1 < contour_2);
}

std::vector<std::vector<cv::Point>> locate_plates(cv::Mat &frame)
{
    cv::Mat processed = frame;
    cv::resize(frame, processed, cv::Size(512, 512));

    if (frame.channels() == 3)
    {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
    }

    cv::Mat black_hat;
    cv::Mat rect_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    cv::morphologyEx(processed, black_hat, cv::MORPH_BLACKHAT, rect_kernel);

    cv::Mat white_hat;
    cv::Mat square_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(processed, white_hat, cv::MORPH_CLOSE, square_kernel);
    cv::threshold(white_hat, white_hat, 0, 255, cv::THRESH_OTSU);

    cv::Mat X;
    double min, max;
    int dx = 1, dy = 0, ddepth = CV_32F, k_size = -1;
    cv::Sobel(black_hat, X, ddepth, dx, dy, k_size);
    X = cv::abs(X);
    cv::minMaxLoc(X, &min, &max);
    X = 255 * ((X - min) / (max - min));
    X.convertTo(X, CV_8U);

    cv::GaussianBlur(X, X, cv::Size(5, 5), 0);
    cv::morphologyEx(X, X, cv::MORPH_CLOSE, rect_kernel);
    cv::threshold(X, X, 0, 255, cv::THRESH_OTSU);

    cv::erode(X, X, 2);
    cv::dilate(X, X, 2);

    cv::bitwise_and(X, X, white_hat);
    cv::dilate(X, X, 2);
    cv::erode(X, X, 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(X, contours, cv::noArray(), cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(), compare_contours);
    std::vector<std::vector<cv::Point>> top_contours;
    top_contours.assign(contours.end() - 5, contours.end());

    return top_contours;
}

int main(int argc, const char **argv)
{
    cv::Mat img = cv::imread("car.jpg", cv::IMREAD_COLOR);

    if (!img.empty())
    {
        std::vector<std::vector<cv::Point>> candidates = locate_plates(img);

        const int width = img.cols;
        const int height = img.rows;
        const float ratio_width = width / (float)512;
        const float ratio_height = height / (float)512;

        std::vector<cv::Rect> ROI;
        for (std::vector<cv::Point> current : candidates)
        {
            cv::Rect temp = cv::boundingRect(current);
            float difference = temp.area() - cv::contourArea(current);
            if (difference < 2000)
            {
                ROI.push_back(temp);
            }
        }

        ROI.erase(std::remove_if(ROI.begin(), ROI.end(),
                                 [](cv::Rect temp)
                                 {
                                     const float aspect_ratio =
                                         temp.width / (float)temp.height;
                                     return aspect_ratio < 1 || aspect_ratio > 6;
                                 }),
                  ROI.end());

        cv::RNG R(12345);

        for (cv::Rect rectangle : ROI)
        {
            cv::Scalar color =
                cv::Scalar(R.uniform(0, 255), R.uniform(0, 255), R.uniform(0, 255));
            cv::rectangle(
                img, cv::Point(rectangle.x * ratio_width, rectangle.y * ratio_height),
                cv::Point((rectangle.x + rectangle.width) * ratio_width,
                          (rectangle.y + rectangle.height) * ratio_height),
                color, 3, cv::LINE_8, 0);
        }

        std::vector<cv::Mat> plates;
        for (cv::Rect area : ROI)
        {
            plates.push_back(
                img(cv::Rect(area.x * ratio_width, area.y * ratio_height,
                             area.width * ratio_width, area.height * ratio_height)));
        }

        tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();

        ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
        ocr->SetPageSegMode(tesseract::PSM_AUTO);

        std::vector<std::string> plate_texts;
        for (cv::Mat plate : plates)
        {
            ocr->SetImage(plate.data, plate.cols, plate.rows, 3, plate.step);

            plate_texts.push_back(ocr->GetUTF8Text());
        }

        ocr->End();

        cv::imwrite("cropped.jpg", plates[0]);

        std::cout << "Plate text:" << std::endl;
        for (std::string plate_text : plate_texts)
        {
            std::cout << plate_text << "\n";
        }
        std::cout << std::endl;

        return 0;
    }
    else
    {
        std::cout << "Cannot open image!" << std::endl;
        return -1;
    }
}
