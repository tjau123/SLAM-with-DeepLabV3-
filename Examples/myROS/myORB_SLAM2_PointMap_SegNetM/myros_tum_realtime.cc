/*
 *--------------------------------------------------------------------------------------------------
 hjx-2022-04-05

 *--------------------------------------------------------------------------------------------------
 */

#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include <octomap/octomap.h>    
#include <octomap/ColorOcTree.h>
 
#include <../../../include/mySystem.h>

using namespace octomap;
using namespace std;
ros::Publisher CamPose_Pub;
ros::Publisher Camodom_Pub;
ros::Publisher odom_pub;

geometry_msgs::PoseStamped Cam_Pose;
geometry_msgs::PoseWithCovarianceStamped Cam_odom;

cv::Mat Camera_Pose;
tf::Transform orb_slam;
tf::TransformBroadcaster * orb_slam_broadcaster;
std::vector<float> Pose_quat(4);
std::vector<float> Pose_trans(3);

ros::Time current_time, last_time;
double lastx=0,lasty=0,lastth=0;
unsigned int a =0,b=0; 
octomap::ColorOcTree tree( 0.05 );

void Pub_CamPose(cv::Mat &pose);

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);
typedef octomap::ColorOcTree::leaf_iterator it_t;

string a1="Vocabulary/ORBvoc.txt";
string a2="Examples/myROS/myORB_SLAM2_PointMap_SegNetM/TUM3.yaml";
string a3="/home/hjx/datasets/rgbd_dataset_freiburg3_walking_xyz";
string a4= "/home/hjx/datasets/rgbd_dataset_freiburg3_walking_xyz/associations.txt";

//string a3="/home/hjx/datasets/rgbd_dataset_freiburg2_desk_with_person";
//string a4= "/home/hjx/datasets/rgbd_dataset_freiburg2_desk_with_person/associations.txt";

string a5="Examples/myROS/myORB_SLAM2_PointMap_SegNetM/prototxts/segnet_pascal.prototxt";
//string a6="/home/hjx/pb/frozen_inference_graph.pbpython";
string a6="/home/hjx/models-master/research/deeplab/exp/voc_train/train1025/export/frozen_inference_graph.pb";

string a7="Examples/myROS/myORB_SLAM2_PointMap_SegNetM/tools/pascal.png";

int main(int argc, char **argv)
{
    cout<<"-----------------"<<a3<< "----------------" <<endl;

    ros::init(argc, argv, "myTUM");
    ros::start();


    //cv::Mat imgrgb1 = cv::imread("/home/hjx/datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.592026.png",CV_LOAD_IMAGE_UNCHANGED);

//    ::google::InitGoogleLogging(argv[0]);
   
//    if(argc != 7)
//    {
//        cerr << endl << "Usage: myTUM path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_prototxt path_to_caffemodel path_to_pascal.png" << endl;
//        return 1;
//    }

    // Retrieve paths to images 检索图像的路径
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    //vector<string> vstrImageFilenamesS;
    vector<double> vTimestamps;
    string strAssociationFilename = string(a4);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    
    // Check consistency in the number of images and depthmaps检查图像和深度图数量的一致性
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::Viewer *viewer;
    viewer = new ORB_SLAM2::Viewer();
    ORB_SLAM2::mySystem SLAM(a1,a2, a3,a6,a7,ORB_SLAM2::mySystem::RGBD, viewer);
    usleep(50);
    // Vector for tracking time statistics用于跟踪时间统计的向量
    vector<double> vTimesTrack;
    vTimesTrack.resize(nImages);
    vector<double> vOrbTime;
    vOrbTime.resize(nImages);
    vector<double> vMovingTime;
    vMovingTime.resize(nImages);
    double segmentationTime=0;
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    ros::Rate loop_rate(50);
    ros::NodeHandle nh;
    
    CamPose_Pub = nh.advertise<geometry_msgs::PoseStamped>("/Camera_Pose",1);
    Camodom_Pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/Camera_Odom", 1);
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);

    current_time = ros::Time::now();
    last_time = ros::Time::now();
    int ni=0;
    while(ros::ok()&&ni<nImages)
    {
        //string aaa=string(a3)+"/"+vstrImageFilenamesRGB[ni];
        //cout << aaa<<endl;
        imRGB = cv::imread(string(a3)+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(a3)+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(a3) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
	    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        Camera_Pose =  SLAM.TrackRGBD(imRGB,imD,tframe);


//        cv::imshow("predict",Camera_Pose);
//        cv::waitKey(0);// 按任意键在0秒后退出窗口，不写这句话是不会显示出窗口的

        std::cout<<"*******************************************************8"<<endl;

	    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
	    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        cout << "SLAM TrackRGBD all time =" << ttrack*1000 << endl << endl;

	    double orbTimeTmp=SLAM.mpTracker->orbExtractTime;
	    double movingTimeTmp=SLAM.mpTracker->movingDetectTime;
	    segmentationTime=SLAM.mpSegment->mSegmentTime;
	    Pub_CamPose(Camera_Pose); 
        vTimesTrack[ni]=ttrack;	
	    vOrbTime[ni]=orbTimeTmp;
 	    vMovingTime[ni]=movingTimeTmp;
        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
	    {
            usleep((T-ttrack)*1e6);
	    }
	    ni++;
        ros::spinOnce();
	    loop_rate.sleep();
    }
    // Stop all threads
    SLAM.Shutdown();
    
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    double totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    double orbTotalTime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        orbTotalTime+=vOrbTime[ni];
    }
    double movingTotalTime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        movingTotalTime+=vMovingTime[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    cout << "mean orb extract time =" << orbTotalTime/nImages <<  endl;
    cout << "mean moving detection time =" << movingTotalTime/nImages<<  endl;
    cout << "mean segmentation time =" << segmentationTime/nImages<<  endl;
   
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");	
    
    ros::shutdown();
    return 0;
}

void Pub_CamPose(cv::Mat &pose)
{
    cv::Mat Rwc(3,3,CV_32F);
	cv::Mat twc(3,1,CV_32F);
	Eigen::Matrix<double,3,3> rotationMat;
	orb_slam_broadcaster = new tf::TransformBroadcaster;
	if(pose.dims<2 || pose.rows < 3)
	{
        Rwc = Rwc;
		twc = twc;
	}
	else
	{
		Rwc = pose.rowRange(0,3).colRange(0,3).t();
		twc = -Rwc*pose.rowRange(0,3).col(3);
		
		rotationMat << Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2),
					Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2),
					Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2);
		Eigen::Quaterniond Q(rotationMat);

		Pose_quat[0] = Q.x(); Pose_quat[1] = Q.y();
		Pose_quat[2] = Q.z(); Pose_quat[3] = Q.w();
		
		Pose_trans[0] = twc.at<float>(0);
		Pose_trans[1] = twc.at<float>(1);
		Pose_trans[2] = twc.at<float>(2);
		
		orb_slam.setOrigin(tf::Vector3(Pose_trans[2], -Pose_trans[0], -Pose_trans[1]));
		orb_slam.setRotation(tf::Quaternion(Q.z(), -Q.x(), -Q.y(), Q.w()));
		orb_slam_broadcaster->sendTransform(tf::StampedTransform(orb_slam, ros::Time::now(), "/map", "/base_link"));
		
		Cam_Pose.header.stamp = ros::Time::now();
		Cam_Pose.header.frame_id = "/map";
		tf::pointTFToMsg(orb_slam.getOrigin(), Cam_Pose.pose.position);
		tf::quaternionTFToMsg(orb_slam.getRotation(), Cam_Pose.pose.orientation);
		
		Cam_odom.header.stamp = ros::Time::now();
		Cam_odom.header.frame_id = "/map";
		tf::pointTFToMsg(orb_slam.getOrigin(), Cam_odom.pose.pose.position);
		tf::quaternionTFToMsg(orb_slam.getRotation(), Cam_odom.pose.pose.orientation);
		Cam_odom.pose.covariance = {0.01, 0, 0, 0, 0, 0,
									0, 0.01, 0, 0, 0, 0,
									0, 0, 0.01, 0, 0, 0,
									0, 0, 0, 0.01, 0, 0,
									0, 0, 0, 0, 0.01, 0,
									0, 0, 0, 0, 0, 0.01};
		
		CamPose_Pub.publish(Cam_Pose);
		Camodom_Pub.publish(Cam_odom);
		
		nav_msgs::Odometry odom;
		odom.header.stamp =ros::Time::now();
		odom.header.frame_id = "/map";

		// Set the position
		odom.pose.pose.position = Cam_odom.pose.pose.position;
		odom.pose.pose.orientation = Cam_odom.pose.pose.orientation;

		// Set the velocity
		odom.child_frame_id = "/base_link";
		current_time = ros::Time::now();
		double dt = (current_time - last_time).toSec();
		double vx = (Cam_odom.pose.pose.position.x - lastx)/dt;
		double vy = (Cam_odom.pose.pose.position.y - lasty)/dt;
		double vth = (Cam_odom.pose.pose.orientation.z - lastth)/dt;
		
		odom.twist.twist.linear.x = vx;
		odom.twist.twist.linear.y = vy;
		odom.twist.twist.angular.z = vth;

		// Publish the message
		odom_pub.publish(odom);
		
		last_time = current_time;
		lastx = Cam_odom.pose.pose.position.x;
		lasty = Cam_odom.pose.pose.position.y;
		lastth = Cam_odom.pose.pose.orientation.z;
	}
}
//可复用
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;

            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}


