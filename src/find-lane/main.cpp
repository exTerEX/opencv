#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <vector>

class FindLane
{
private:
    int min_vote, shift_v, shift_h;
    double min_length, max_gap;

    cv::Mat img;
    std::vector<cv::Vec4i> lines;

public:
    FindLane() : min_vote(15), min_length(5), max_gap(80), shift_v(0), shift_h(0) {}

    ~FindLane() = default;

    void set_min_vote(int vote) { min_vote = vote; }

    void set_line_properties(double length, double gap)
    {
        min_length = length;
        max_gap = gap;
    }

    void set_shift(int v, int h)
    {
        shift_v = v;
        shift_h = h;
    }

    void draw_detected_lines(cv::Mat &img, cv::Scalar color)
    {
        std::vector<cv::Vec4i>::const_iterator iterator = lines.begin();

        while (iterator != lines.end())
        {
            cv::Point p1((*iterator)[0] + shift_h, (*iterator)[1] + shift_v);
            cv::Point p2((*iterator)[2] + shift_h, (*iterator)[3] + shift_v);

            // Remove the shortest lines
            double line_length = sqrt(exp2(p2.x - p1.x) + exp2(p2.y - p1.y));
            if (line_length >= 1.0e+30)
            {
                cv::line(img, p1, p2, color, 3);
            }

            ++iterator;
        }
    }

    std::vector<cv::Vec4i> find_lines(cv::Mat &b)
    {
        lines.clear();
        cv::HoughLinesP(b, lines, 1, M_PI / 180, min_vote, min_length, max_gap);

        return lines;
    }
};

bool debug = true;

int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("assets/data/lane.jpg", cv::IMREAD_COLOR);

    FindLane obj;

    cv::Mat contours;
    std::vector<cv::Vec2f> lines_2f;
    std::vector<cv::Vec4i> lines_4i;
    std::vector<cv::Vec2f>::const_iterator iterator;

    if (!img.empty())
    {
        cv::Canny(img, contours, 125, 245);

        if (debug)
        {
            cv::imwrite("output/contours.jpg", contours);
        }

        int threshold = 125;
        if (threshold < 1 or lines_2f.size() > 2)
        {
            threshold = 300;
        }
        else
        {
            threshold += 25;
        }

        while (lines_2f.size() < 4 && threshold > 0)
        {
            HoughLines(contours, lines_2f, 1, M_PI / 180, threshold);
            threshold -= 5;
        }

        cv::Mat result(img.size(), CV_8U, cv::Scalar(255));
        img.copyTo(result);

        // Draw lines
        cv::Mat hough(img.size(), CV_8U, cv::Scalar(0));

        iterator = lines_2f.begin();
        while (iterator != lines_2f.end())
        {
            float rho = (*iterator)[0];
            float theta = (*iterator)[1];

            if ((theta > 0.09 && theta < 1.48) || (theta < 3.14 && theta > 1.66))
            {
                cv::Point pt1(rho / std::cos(theta), 0);
                cv::Point pt2((rho - result.rows * std::sin(theta)) / std::cos(theta), result.rows);

                cv::line(result, pt1, pt2, cv::Scalar(255, 255, 255), 1);
                cv::line(hough, pt1, pt2, cv::Scalar(255, 255, 255), 1);
            }

            ++iterator;
        }

        if (debug)
        {
            cv::imwrite("output/hough.jpg", result);
        }

        // Detect lines
        lines_4i = obj.find_lines(contours);
        cv::Mat hough_p(img.size(), CV_8U, cv::Scalar(0));

        obj.draw_detected_lines(hough_p, cv::Scalar(0));

        if (debug)
        {
            cv::imwrite("output/hough_p.jpg", hough_p);
        }

        cv::bitwise_and(hough_p, hough, hough_p);
        cv::Mat hough_p_inv(img.size(), CV_8U, cv::Scalar(0));
        cv::threshold(hough_p, hough_p_inv, 150, 255, cv::THRESH_BINARY_INV);

        if (debug)
        {
            cv::imwrite("output/detect_houghP.jpg", hough_p);
        }

        cv::Canny(hough_p_inv, contours, 100, 255);
        lines_4i = obj.find_lines(contours);

        obj.set_line_properties(5, 2);
        obj.set_min_vote(1);

        obj.draw_detected_lines(img, cv::Scalar(0));

        cv::imwrite("output/lines.jpg", img);

        lines_2f.clear();
    }
}
