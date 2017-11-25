#include <iostream>
#include <fstream>

#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/keypoints/iss_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace std;
//using namespace pcl;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
//typedef pcl::PointCloud<pcl::Normal> Normal;

typedef pcl::FPFHSignature33 FPFHT;
typedef pcl::PointCloud<FPFHT> FPFHCloud;
const float  VOXEL_GRID_SIZE = 0.001;
//关键点提取
const float  min_neighbors=5;
const float  iss_resolution=0.0005;//
//法向量和特征
const double radius_normal=10;
const double radius_feature=10;
//错误点对排除
const double epsilon=0.01;//错误点对排除，判断内点的标准
const double N=2000;//错误点对排除，最大迭代次数
//sac_ia粗配准
const double n=15;//每次随机选择的样本数
const double min_sample_dist=0.01;//样本最小间距
const double k=5;//特征球选前k个
const double max_correspondence_dist=1000;
const double max_sacis_iteration=1000;

pcl::console::TicToc timecal;
pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

void voxelFilter(PointCloud::Ptr &cloud_in, PointCloud::Ptr &cloud_out, float gridsize){
	pcl::VoxelGrid<PointT> vox_grid;
	vox_grid.setLeafSize(gridsize, gridsize, gridsize);
	vox_grid.setInputCloud(cloud_in);
	vox_grid.filter(*cloud_out);
	return;
}

PointCloud::Ptr getKeypoints(PointCloud::Ptr cloud,double resolution)
{
  PointCloud::Ptr keypoints (new PointCloud);
  pcl::ISSKeypoint3D<PointT,PointT> iss_detector;
  iss_detector.setSearchMethod(tree);
  iss_detector.setSalientRadius(6*resolution);
  iss_detector.setNonMaxRadius(4*resolution);
  iss_detector.setThreshold21(0.975);
  iss_detector.setThreshold32(0.975);
  iss_detector.setMinNeighbors(min_neighbors);
  iss_detector.setNumberOfThreads(4);
  iss_detector.setInputCloud(cloud);
  iss_detector.compute(*keypoints);
  cout<<"After keypoints detecting, left: "<<keypoints->points.size()<<"data points"<<endl;

  return keypoints;

}

pcl::PointCloud<pcl::Normal>::Ptr getNormals(PointCloud::Ptr cloud, double radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normalsPtr (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointT,pcl::Normal> norm_est;
    norm_est.setInputCloud(cloud);
    norm_est.setRadiusSearch(radius);
    norm_est.compute(*normalsPtr);
    return normalsPtr;

}

FPFHCloud::Ptr getFeatures(PointCloud::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normals,double radius)
{
    FPFHCloud::Ptr features (new FPFHCloud);
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::FPFHEstimationOMP<PointT,pcl::Normal,FPFHT> fpfh_est;
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setNumberOfThreads(4);
    fpfh_est.setSearchMethod(tree);
    fpfh_est.setRadiusSearch(radius);
    fpfh_est.compute(*features);
    return features;
}

Eigen::Matrix4f sac_ia_align(PointCloud::Ptr source,PointCloud::Ptr target,
                             FPFHCloud::Ptr source_feature,FPFHCloud::Ptr target_feature,
                             int max_sacia_iterations,double min_sample_dist,double max_correspondence_dist)
{
    pcl::SampleConsensusInitialAlignment<PointT,PointT,FPFHT> sac_ia;
    Eigen::Matrix4f final_transformation;
    sac_ia.setInputSource(target);
    sac_ia.setInputTarget(source);
    sac_ia.setSourceFeatures(target_feature);    
    sac_ia.setTargetFeatures(source_feature);
    sac_ia.setNumberOfSamples(n);
    sac_ia.setMinSampleDistance(min_sample_dist);
    sac_ia.setCorrespondenceRandomness(k);
    sac_ia.setMaxCorrespondenceDistance(max_correspondence_dist);
    sac_ia.setMaximumIterations(max_sacia_iterations);    
    
    PointCloud::Ptr finalcloud (new PointCloud);
    timecal.tic();
    sac_ia.align(*finalcloud);
    cout<<"采用SAC_IA粗配准用了："<<timecal.toc()<<"ms"<<endl;
    final_transformation=sac_ia.getFinalTransformation();
    return final_transformation;

}
//左右显示变换前后点云对
void viewPair(PointCloud::Ptr cloud1, PointCloud::Ptr cloud2,
	PointCloud::Ptr cloud1al, PointCloud::Ptr cloud2al){

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    //pcl::visualization::PCLVisualizer viewer("3D viewer");
    viewer->initCameraParameters();
	int v1(0), v2(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Before Alignment", 10, 10, "v1 text", v1);
	PointCloudColorHandlerCustom<PointT> green(cloud1, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> red(cloud2, 255, 0, 0);
	viewer->addPointCloud(cloud1, green, "v1_target", v1);
	viewer->addPointCloud(cloud2, red, "v1_sourse", v1);

	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0, 0, 0, v2);
	viewer->addText("After Alignment", 10, 10, "v2 text", v2);
	PointCloudColorHandlerCustom<PointT> green2(cloud1al, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> red2(cloud2al, 255, 0, 0);
	viewer->addPointCloud(cloud1al, green2, "v2_target", v2);
	viewer->addPointCloud(cloud2al, red2, "v2_sourse", v2);
    viewer->spin();   

}


int main(int argc,char** argv)
{
    PointCloud::Ptr source (new PointCloud);
    PointCloud::Ptr target (new PointCloud);
    PointCloud::Ptr result1 (new PointCloud);
    PointCloud::Ptr result2 (new PointCloud);
    if(pcl::io::loadPCDFile(argv[1],*source)<0)
    {
        PCL_ERROR("This dir doesnot exit %s pcd file.\n",argv[1]);
        return(-1);
    }
    if(pcl::io::loadPCDFile(argv[2],*target)<0)
    {
        PCL_ERROR("This dir doesnot exit %s pcd file.\n",argv[2]);
        return(-1);
    }
    vector<int> indices1;
    vector<int> indices2;
    pcl::removeNaNFromPointCloud(*source,*source,indices1);
    pcl::removeNaNFromPointCloud(*target,*target,indices2);
    PointCloud::Ptr sourceds (new PointCloud);
    PointCloud::Ptr targetds (new PointCloud);    
//降采样
    voxelFilter(source,sourceds,VOXEL_GRID_SIZE);  
    voxelFilter(target,targetds,VOXEL_GRID_SIZE);  
 
//提取关键点
    PointCloud::Ptr sourcekp (new PointCloud);
    PointCloud::Ptr targetkp (new PointCloud);
    sourceds=getKeypoints(sourceds,iss_resolution);
    targetds=getKeypoints(targetds,iss_resolution);

//计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr source_normal (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normal (new pcl::PointCloud<pcl::Normal>);
    source_normal=getNormals(sourceds,radius_normal);
    target_normal=getNormals(targetds,radius_normal);
   
//计算FPFH特征    
    FPFHCloud::Ptr source_feature (new FPFHCloud);
    FPFHCloud::Ptr target_feature (new FPFHCloud);
    source_feature=getFeatures(sourceds,source_normal,radius_feature);
    target_feature=getFeatures(targetds,target_normal,radius_feature);
  
//估计点对对应关系
    pcl::CorrespondencesPtr corres(new pcl::Correspondences);
    pcl::registration::CorrespondenceEstimation<FPFHT,FPFHT> corres_est;
    corres_est.setInputSource(source_feature);
    corres_est.setInputTarget(target_feature);
    corres_est.determineReciprocalCorrespondences(*corres);


//错误点对排除    
    pcl::Correspondences inliers_corres;
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> sac;
    sac.setInputSource(sourceds);
    sac.setInputTarget(targetds);
    sac.setInputCorrespondences(corres);
    sac.setInlierThreshold(epsilon);
    sac.setMaxIterations(N);    
    sac.getCorrespondences(inliers_corres);
    Eigen::Matrix4f transformation = sac.getBestTransformation();
    cout<<"错误点对排除，内点最多时的变换：\n"<<transformation<<"\n"<<endl;


//估计刚性变换
    Eigen::Matrix4f init_trans1;
    pcl::registration::TransformationEstimationSVD<PointT,PointT> svd;
    svd.estimateRigidTransformation(*sourceds,*targetds,inliers_corres,init_trans1);
    cout<<"一次性估计变换：\n"<<init_trans1.inverse()<<"\n"<<endl;
    pcl::transformPointCloud(*target,*result1,init_trans1.inverse());

//SAC-IA配准
    Eigen::Matrix4f init_trans2;
    init_trans2=sac_ia_align(sourceds,targetds,source_feature,target_feature, max_sacis_iteration,min_sample_dist,max_correspondence_dist);
  
    pcl::transformPointCloud(*target,*result2,init_trans2);    
    cout<<init_trans2<<endl;

    viewPair(source,target,source,result1);
    viewPair(source,target,source,result2);
   
    return 0;






}
