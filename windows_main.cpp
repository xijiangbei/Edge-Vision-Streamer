#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

using namespace std;
using namespace cv;
using namespace cv::dnn; 

struct ObjectResult {
    Rect box;       
    string name;    
};

class VisionThreadPool {
private: 
    mutex mtx_lock;                           
    condition_variable cv_bell;               
    queue<shared_ptr<Mat>> taskQueue_belt;    
    vector<ObjectResult> latestObjects_board; 
    bool isShutDown_flag;                     
    vector<thread> workers_pool;              
    Net aiModel;                              

    const vector<string> classNames = {"Background", "Aeroplane", "Bicycle", "Bird", "Boat",
        "Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Diningtable", "Dog", "Horse",
        "Motorbike", "Person", "Pottedplant", "Sheep", "Sofa", "Train", "Tvmonitor"};

    void backgroundWorkerTask() {
        while (true) {
            shared_ptr<Mat> smartPhoto;
            {
                unique_lock<mutex> lock(mtx_lock);
                cv_bell.wait(lock, [this] { return !taskQueue_belt.empty() || isShutDown_flag; });
                if (isShutDown_flag && taskQueue_belt.empty()) return; 
                smartPhoto = taskQueue_belt.front(); 
                taskQueue_belt.pop();
            } 
            
            vector<ObjectResult> detected_objs;


            if (!aiModel.empty()) {

                Mat blob = blobFromImage(*smartPhoto, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false, false);
                
                aiModel.setInput(blob);        
                Mat output = aiModel.forward(); 

                Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

                for (int i = 0; i < detectionMat.rows; i++) {
                    float confidence = detectionMat.at<float>(i, 2);

                    if (confidence > 0.5) { 
                        int classId = (int)(detectionMat.at<float>(i, 1)); 
                        
                        int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * 640);
                        int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * 480);
                        int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * 640);
                        int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * 480);

                        Rect object_box(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                        
                        string obj_name = classNames[classId] + " " + to_string((int)(confidence * 100)) + "%";

                        detected_objs.push_back({object_box, obj_name}); 
                    }
                }
            } else {

                static bool warned = false;
                if (!warned) { cout << "找不到模型文件" << endl; warned = true; }
            }


            lock_guard<mutex> lock(mtx_lock);
            latestObjects_board = detected_objs; 
        }
    }

public: 
    VisionThreadPool(int threadCount = 2) {
        isShutDown_flag = false;
        
        try {
            aiModel = readNetFromCaffe(
                "D:/code/c/day20/deploy.prototxt", 
                "D:/code/c/day20/mobilenet_iter_73000.caffemodel"
            );
            cout << "模型加载成功！" << endl;
        } catch (const Exception& e) {
            cout << "模型加载失败" << endl;
        }

        for (int i = 0; i < threadCount; i++) {
            workers_pool.push_back(thread(&VisionThreadPool::backgroundWorkerTask, this)); 
        }
    }
    
    ~VisionThreadPool() {
        { lock_guard<mutex> lock(mtx_lock); isShutDown_flag = true; }
        cv_bell.notify_all(); 
        for (auto& t : workers_pool) if (t.joinable()) t.join(); 
    }
    
    void pushFrame(Mat frame) {
        int current_size = 0;
        { lock_guard<mutex> lock(mtx_lock); current_size = taskQueue_belt.size(); }
        if (current_size >= 2) return; 
        
        Mat smallFrame;
        resize(frame, smallFrame, Size(300, 300)); 
        auto smartPhoto = make_shared<Mat>(smallFrame.clone());          
        
        { lock_guard<mutex> lock(mtx_lock); taskQueue_belt.push(smartPhoto); }
        cv_bell.notify_one();                         
    }
    
    vector<ObjectResult> getLatestObjects() {
        lock_guard<mutex> lock(mtx_lock);
        return latestObjects_board; 
    }
};

int main() {
    VideoCapture pc_camera(0); 
    if (!pc_camera.isOpened()) {
        cout << "没找到电脑摄像头" << endl;
        return -1;
    }
 
    pc_camera.set(CAP_PROP_FRAME_WIDTH, 640);
    pc_camera.set(CAP_PROP_FRAME_HEIGHT, 480);

    VisionThreadPool myEngine(2); 
    Mat current_frame;            

    cout << "本地深度学习环境启动成功 按ESC键退出。" << endl;

    while (true) {
        pc_camera >> current_frame; 
        if (current_frame.empty()) break;

        myEngine.pushFrame(current_frame); 

        vector<ObjectResult> objects = myEngine.getLatestObjects(); 

        for (auto& obj : objects) {

            rectangle(current_frame, obj.box.tl(), obj.box.br(), Scalar(0, 255, 255), 2); 

            putText(current_frame, obj.name, Point(obj.box.x, obj.box.y - 8), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        imshow("Deep Learning Monitor", current_frame);

        if (waitKey(30) == 27) break;
    }

    pc_camera.release(); 
    destroyAllWindows(); 
    return 0;
}