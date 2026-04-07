#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

using namespace std;
using namespace cv;
using namespace cv::dnn; // 【新增】引入 OpenCV 深度学习专属武器库

// ==========================================================
// 【大厂数据结构】用来装 AI 识别出来的物体信息（包裹）
// ==========================================================
struct ObjectResult {
    Rect box;       // 【物理外框】物体在屏幕上的坐标和大小
    string name;    // 【身份铭牌】物体的英文名 + 确信度，比如 "Dog 95%"
};

// ================= 深度学习 AI 算力车间 =================
class VisionThreadPool {
private: 
    mutex mtx_lock;                           // 【车间大门锁】防止画面撕裂
    condition_variable cv_bell;               // 【叫醒铃声】没活干就睡觉，省 CPU
    queue<shared_ptr<Mat>> taskQueue_belt;    // 【图片传送带】缓冲摄像头传来的画面
    vector<ObjectResult> latestObjects_board; // 【全新公告板】不仅存框，还存名字
    bool isShutDown_flag;                     // 【下班标志】
    vector<thread> workers_pool;              // 【工位池】
    Net aiModel;                              // 【最强大脑】注意！这里换成了 DNN 深度网络核心

    // 【AI 的英语词典】这 20 个单词，就是这个模型脑子里唯一认识的 20 种东西
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

            // 【防御补丁】先检查脑子里有没有装载模型，没带书绝对不考试！
            if (!aiModel.empty()) {
                // 【深度学习核心 1：喂食预处理】
                // 深度学习模型很挑食，必须把图片强行拉伸成 300x300，并且减去 127.5 的均值，乘以 0.007843 的比例
                // 这行代码会把普通图片变成 AI 爱吃的高维矩阵 (Blob)
                Mat blob = blobFromImage(*smartPhoto, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false, false);
                
                // 【深度学习核心 2：开始脑内推演】
                aiModel.setInput(blob);         // 喂到嘴里
                Mat output = aiModel.forward(); // AI 咽下去并疯狂计算，拉出结果 (output)

                // 【深度学习核心 3：破译密码】
                // AI 吐出的是一个人类看不懂的四维矩阵，我们把它拍扁成 2 维表格
                Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

                // 遍历表格的每一行（每一行代表 AI 发现的一个疑似物体）
                for (int i = 0; i < detectionMat.rows; i++) {
                    float confidence = detectionMat.at<float>(i, 2); // 第 3 列(索引2)存着 AI 的自信心
                    
                    // 【过滤门槛】自信心超过 50% (0.5)，我们才承认它是真的
                    if (confidence > 0.5) { 
                        int classId = (int)(detectionMat.at<float>(i, 1)); // 第 2 列(索引1)存着物体的学号
                        
                        // 【算盘时刻】AI 给出的坐标是 0~1 的百分比，必须乘以 640 和 480，还原回电脑屏幕的真实像素位置
                        int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * 640);
                        int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * 480);
                        int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * 640);
                        int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * 480);

                        Rect object_box(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                        
                        // 【组装名字标签】拿学号去查词典找英文名，再加上百分比确信度。比如："Person 98%"
                        string obj_name = classNames[classId] + " " + to_string((int)(confidence * 100)) + "%";

                        detected_objs.push_back({object_box, obj_name}); // 打包塞进结果集
                    }
                }
            } else {
                // 如果模型没加载成功，只报一次警，绝不让程序死机
                static bool warned = false;
                if (!warned) { cout << "[致命警告] 找不到 MobileNet 模型文件，AI 拒绝工作！" << endl; warned = true; }
            }

            // 【下班交接】算完了，把包含名字和框的数据包，挂在黑板上给主线程看
            lock_guard<mutex> lock(mtx_lock);
            latestObjects_board = detected_objs; 
        }
    }

public: 
    VisionThreadPool(int threadCount = 2) {
        isShutDown_flag = false;
        
        // 【加载深度学习模型】这里用的是你刚才下载的那两个真实文件名！
        try {
            aiModel = readNetFromCaffe(
                "D:/code/c/day20/deploy.prototxt", 
                "D:/code/c/day20/mobilenet_iter_73000.caffemodel"
            );
            cout << "[系统] 深度学习大模型 (MobileNet-SSD) 加载成功！" << endl;
        } catch (const Exception& e) {
            cout << "[系统] 路径不对或文件损坏，模型加载失败！" << endl;
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
        if (current_size >= 2) return; // 【防卡死】图太多了直接丢掉
        
        // 深度学习模型必须看彩色图！所以我们不再转黑白，只是把图缩小到 300x300 减轻队列内存压力
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

// ================= 主线程：本地 Windows 傻瓜式看图 =================
int main() {
    VideoCapture pc_camera(0); // 打开电脑默认摄像头
    if (!pc_camera.isOpened()) {
        cout << "[系统] 没找到电脑摄像头，请检查！" << endl;
        return -1;
    }
    // 【强制统一分辨率】强制把画面锁死在 640x480，确保刚才 AI 算盘那里的乘法坐标准确无误！
    pc_camera.set(CAP_PROP_FRAME_WIDTH, 640);
    pc_camera.set(CAP_PROP_FRAME_HEIGHT, 480);

    VisionThreadPool myEngine(2); // 雇佣 2 个深度学习工程师
    Mat current_frame;            

    cout << "[系统] Windows 本地深度学习环境启动成功！按 ESC 键退出。" << endl;

    while (true) {
        pc_camera >> current_frame; 
        if (current_frame.empty()) break;

        // 1. 把拍到的彩图扔给后厨
        myEngine.pushFrame(current_frame); 
        
        // 2. 从后厨拿取写好“名字和框”的数据包裹
        vector<ObjectResult> objects = myEngine.getLatestObjects(); 

        // 3. 【给万物画框 + 贴名字标签】
        for (auto& obj : objects) {
            // 画一个极具安防逼格的黄色细框
            rectangle(current_frame, obj.box.tl(), obj.box.br(), Scalar(0, 255, 255), 2); 
            // 在框的左上角，用高亮的绿色字体写上物体的英文名字和百分比
            putText(current_frame, obj.name, Point(obj.box.x, obj.box.y - 8), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // 直接在电脑屏幕上弹窗展示
        imshow("Deep Learning Monitor", current_frame);

        // 每秒刷新，按键盘左上角的 ESC 键退出
        if (waitKey(30) == 27) break;
    }

    pc_camera.release(); 
    destroyAllWindows(); 
    return 0;
}