#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>
#include <string>

// Linux 网络与底层相关的头文件
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define MAX_EVENTS 1024
#define PORT 8080


struct ObjectResult { Rect box; string name; };

class VisionThreadPool {
private: 
    mutex mtx_lock;                           
    condition_variable cv_bell;               
    queue<shared_ptr<Mat>> taskQueue_belt;    
    vector<ObjectResult> latestObjects_board; 
    bool isShutDown_flag;                     
    vector<thread> workers_pool;              
    Net aiModel;                              
    const vector<string> classNames = {"Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Diningtable", "Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep", "Sofa", "Train", "Tvmonitor"};

    void backgroundWorkerTask() {
        while (true) {
            shared_ptr<Mat> smartPhoto;
            {
                unique_lock<mutex> lock(mtx_lock);
                cv_bell.wait(lock, [this] { return !taskQueue_belt.empty() || isShutDown_flag; });
                if (isShutDown_flag && taskQueue_belt.empty()) return; 
                smartPhoto = taskQueue_belt.front(); taskQueue_belt.pop();
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
                        detected_objs.push_back({Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom), classNames[classId] + " " + to_string((int)(confidence * 100)) + "%"});
                    }
                }
            }
            lock_guard<mutex> lock(mtx_lock);
            latestObjects_board = detected_objs; 
        }
    }

public: 
    VisionThreadPool(int threadCount = 2) {
        isShutDown_flag = false;
        // Linux 端的相对路径加载模型
        aiModel = readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel");
        for (int i = 0; i < threadCount; i++) workers_pool.push_back(thread(&VisionThreadPool::backgroundWorkerTask, this)); 
    }
    ~VisionThreadPool() {
        { lock_guard<mutex> lock(mtx_lock); isShutDown_flag = true; }
        cv_bell.notify_all(); 
        for (auto& t : workers_pool) if (t.joinable()) t.join(); 
    }
    void pushFrame(Mat frame) {
        int current_size = 0;
        { lock_guard<mutex> lock(mtx_lock); current_size = taskQueue_belt.size(); }
        if (current_size >= 2) return; // 【高频考点：主动丢帧防 OOM】
        Mat smallFrame; resize(frame, smallFrame, Size(300, 300)); 
        auto smartPhoto = make_shared<Mat>(smallFrame.clone());          
        { lock_guard<mutex> lock(mtx_lock); taskQueue_belt.push(smartPhoto); }
        cv_bell.notify_one();                         
    }
    vector<ObjectResult> getLatestObjects() {
        lock_guard<mutex> lock(mtx_lock);
        return latestObjects_board; 
    }
};


void setNonBlocking(int sock) {
    int opts = fcntl(sock, F_GETFL);
    if (opts < 0) { perror("fcntl(F_GETFL)"); exit(1); }
    opts = (opts | O_NONBLOCK);
    if (fcntl(sock, F_SETFL, opts) < 0) { perror("fcntl(F_SETFL)"); exit(1); }
}


int main() {

    VideoCapture camera(0, CAP_V4L2);
    if (!camera.isOpened()) { cout << "Failed to open Linux Camera!" << endl; return -1; }
    camera.set(CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CAP_PROP_FRAME_HEIGHT, 480);

    VisionThreadPool myEngine(2);
    Mat frame;
    vector<uchar> jpeg_buffer; 

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    setNonBlocking(server_fd);

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 128);

    int epoll_fd = epoll_create1(0);
    struct epoll_event ev, events[MAX_EVENTS];
    ev.events = EPOLLIN | EPOLLET; 
    ev.data.fd = server_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev);

    cout << "服务器已启动，监听端口: " << PORT << endl;


    while (true) {

        camera >> frame;
        if (frame.empty()) continue;
        myEngine.pushFrame(frame); 
        vector<ObjectResult> objects = myEngine.getLatestObjects(); 
        for (auto& obj : objects) {
            rectangle(frame, obj.box.tl(), obj.box.br(), Scalar(0, 255, 255), 2); 
            putText(frame, obj.name, Point(obj.box.x, obj.box.y - 8), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        imencode(".jpg", frame, jpeg_buffer);


        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, 0); 
        
        for (int i = 0; i < nfds; ++i) {
            if (events[i].data.fd == server_fd) {
                struct sockaddr_in client_addr;
                socklen_t client_addr_len = sizeof(client_addr);
                int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len);
                if (client_fd > 0) {
                    setNonBlocking(client_fd);
                    ev.events = EPOLLOUT | EPOLLET; // 注册可写事件
                    ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);

                    string http_header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
                    send(client_fd, http_header.c_str(), http_header.length(), 0);
                }
            } else if (events[i].events & EPOLLOUT) {

                int client_fd = events[i].data.fd;
                string frame_header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + to_string(jpeg_buffer.size()) + "\r\n\r\n";

                send(client_fd, frame_header.c_str(), frame_header.length(), MSG_NOSIGNAL);
                int bytes_sent = send(client_fd, jpeg_buffer.data(), jpeg_buffer.size(), MSG_NOSIGNAL);
                send(client_fd, "\r\n", 2, MSG_NOSIGNAL);

                if (bytes_sent <= 0) {
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client_fd, NULL);
                    close(client_fd);
                }
            }
        }
    }
    close(server_fd);
    return 0;
}