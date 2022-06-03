#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <string>
#include <vector>

static void help(const char **argv)
{
    std::cout << "\nThis is a simple program to detect faces + eyes of people in "
                 "images.\n"
                 "Usage: "
              << argv[0] << "\n"
              << "   [--face-cascade=<cascade_path>]\n"
                 "   [--eye-cascade=<cascade_path>]\n"
                 "   [--]\n"
                 "   [outfile]\n"
                 "   [infile]\n\n"
                 "\tUsing OpenCV version "
              << CV_VERSION << "\n"
              << std::endl;
}

int main(int argc, const char **argv)
{
    cv::CommandLineParser parser(
        argc, argv,
        "{help h||}"
        "{face-cascade|assets/model/haarcascade_frontalface_alt.xml|}"
        "{eye-cascade|assets/model/haarcascade_eye_tree_eyeglasses.xml|}"
        "{@outfile||}"
        "{@infile||}");

    // Show help text in terminal
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    // Load haar cascades
    cv::CascadeClassifier faceCascade(parser.get<std::string>("face-cascade"));
    cv::CascadeClassifier eyeCascade(parser.get<std::string>("eye-cascade"));

    // Define input and output file names
    std::string outfile = parser.get<std::string>("@outfile");
    std::string infile = parser.get<std::string>("@infile");

    // Load input image
    cv::Mat img = cv::imread(infile, cv::IMREAD_COLOR);

    const static cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 128, 255), cv::Scalar(0, 255, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 255)};

    if (!img.empty())
    {
        double t = 0;
        std::vector<cv::Rect> faces;

        std::cout << "Starting face detection..." << std::endl;

        t = (double)cv::getTickCount();
        faceCascade.detectMultiScale(img, faces, 1.1, 2,
                                     0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        t = (double)cv::getTickCount() - t;

        std::cout << "Detecting faces in image. Detection time = "
                  << t * 1000 / cv::getTickFrequency() << " ms" << std::endl;

        cv::Mat faceImg;

        // Resize grayscale into small image
        cv::resize(img, faceImg, cv::Size(), 1.0, 1.0, cv::INTER_LINEAR_EXACT);

        // Cycle faces
        for (size_t index = 0; index < faces.size(); index++)
        {
            cv::Rect face = faces[index];
            cv::Scalar color = colors[index % 8];
            std::vector<cv::Rect> eyes;

            // Draw rectangle around
            rectangle(img, cv::Point(cvRound(face.x), cvRound(face.y)),
                      cv::Point(cvRound((face.x + face.width - 1)),
                                cvRound((face.y + face.height - 1))),
                      color, 3, 8, 0);

            // Detect eyes in face
            t = (double)cv::getTickCount();
            eyeCascade.detectMultiScale(faceImg(face), eyes, 1.1, 2,
                                        0 | cv::CASCADE_SCALE_IMAGE, cv::Size());
            t = (double)cv::getTickCount() - t;

            std::cout << "Detecting eyes for person " << index + 1
                      << ". Detection time = " << t * 1000 / cv::getTickFrequency()
                      << " ms" << std::endl;

            cv::Point center;

            // Cycle eyes
            for (size_t jndex = 0; jndex < eyes.size(); jndex++)
            {
                cv::Rect eye = eyes[jndex];

                center.x = cvRound(face.x + eye.x + eye.width * 0.5);
                center.y = cvRound(face.y + eye.y + eye.height * 0.5);

                // Draw circle around eyes
                circle(img, center, cvRound((eye.width + eye.height) * 0.25), color, 5,
                       8, 0);
            }
        }
    }

    return cv::imwrite(outfile, img);
}
