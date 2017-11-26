#include <iostream>
#include <string>
#include <boost/make_shared.hpp>
#include <pcl/console/time.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/keypoints/iss_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>//包含fpfh加速计算的omp多核并行计算

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>

//216行  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
using namespace std;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

typedef pcl::FPFHSignature33 FPFHT;
typedef pcl::PointCloud<FPFHT> FPFHCloud;

float  VOXEL_GRID_SIZE = 0.001;//
//关键点提取
float  min_neighbors=100;
float  iss_resolution=5;//
//法向量和特征
double radius_normal=20;
double radius_feature=20;
//错误点对排除
double epsilon=0.01;//错误点对排除，判断内点的标准
double N=2000;//错误点对排除，最大迭代次数
//sac_ia粗配准
double n=15;//每次随机选择的样本数
double min_sample_dist=0.01;//样本最小间距
double k=5;//特征球选前k个
double max_correspondence_dist=1000;
double max_sacia_iterations=1000;
double icp_iteration=1000;//


pcl::console::TicToc timecal;
pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};


void voxelFilter(PointCloud::Ptr &cloud_in, PointCloud::Ptr &cloud_out, float gridsize){
	pcl::VoxelGrid<PointT> vox_grid;
	vox_grid.setLeafSize(gridsize, gridsize, gridsize);
	vox_grid.setInputCloud(cloud_in);
  vox_grid.filter(*cloud_out);
  cout << "Total point: " << cloud_in->points.size()<< " data points "<<endl;
  cout << "After voxelfiltering, left" << cloud_out->points.size()<< " data points\n"<<endl;
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
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(radius);
    //norm_est.setRadiusSearch(radius);
    norm_est.compute(*normalsPtr);
    return normalsPtr;

}


FPFHCloud::Ptr getFeatures(PointCloud::Ptr cloud,pcl::PointCloud<pcl::Normal>::Ptr normals,double radius)
{
    FPFHCloud::Ptr features (new FPFHCloud);    
    pcl::FPFHEstimationOMP<PointT,pcl::Normal,FPFHT> fpfh_est;
    fpfh_est.setNumberOfThreads(4);
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchMethod(tree);
    fpfh_est.setKSearch(radius);
   // fpfh_est.setRadiusSearch(radius);
    fpfh_est.compute(*features);
    return features;
}


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



////////////////////////////////////////////////////////////////////////////////
/** \brief Load a set of PCD files that we want to register together
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd");
  // Suppose the first argument is the actual test model
  for (int i = 1; i < argc; i++)
  {
    std::string fname = std::string (argv[i]);
    // Needs to be at least 5: .plot
    if (fname.size () <= extension.size ())
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower);

    //check that the argument is a pcd file
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // Load the cloud and saves it into the global list of models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);

      models.push_back (m);
    }
  }
}


//sac_ia配准
void sac_ia_align(PointCloud::Ptr src,PointCloud::Ptr tgt,PointCloud::Ptr finalcloud,Eigen::Matrix4f init_transform,
   int max_sacia_iterations,double min_sample_dist,double max_correspondence_dist)
{

  vector<int> indices1;
  vector<int> indices2;  
  PointCloud::Ptr source(new PointCloud);
  PointCloud::Ptr target(new PointCloud);
  pcl::removeNaNFromPointCloud(*src,*source,indices1);
  pcl::removeNaNFromPointCloud(*tgt,*target,indices2);
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

//SAC-IA配准  
  pcl::SampleConsensusInitialAlignment<PointT,PointT,FPFHT> sac_ia;
  Eigen::Matrix4f final_transformation;
  sac_ia.setInputSource(targetds);
  sac_ia.setSourceFeatures(target_feature);
  sac_ia.setInputTarget(sourceds);
  sac_ia.setTargetFeatures(source_feature);
  sac_ia.setMaximumIterations(max_sacia_iterations);
  sac_ia.setMinSampleDistance(min_sample_dist);
  sac_ia.setMaxCorrespondenceDistance(max_correspondence_dist);
  
  PointCloud::Ptr output (new PointCloud);
  timecal.tic();  
  sac_ia.align(*output);
  cout<<"初配准用了 "<<timecal.toc()<<"ms\n"<<endl;
  init_transform=sac_ia.getFinalTransformation();
  pcl::transformPointCloud(*tgt,*finalcloud,init_transform);
  cout<<"初配准变换：\n"<<init_transform<<endl;
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.05, 0.05, 0.05);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    *src = *cloud_src;
    *tgt = *cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-8);
  //reg.setEuclideanFitnessEpsilon(1);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (3.0);  
  //reg.RANSACOutlierRejectionThreshold(1.5); 
  reg.setMaximumIterations (icp_iteration); 
  reg.setEuclideanFitnessEpsilon(1e-8);
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;

  for (int i = 0; i < 1; ++i)
  {    
    points_with_normals_src = reg_result;// save cloud for visualization purpose
    reg.setInputSource (points_with_normals_src);
    timecal.tic();
    reg.align (*reg_result);	
    //cout<<"Applied %ICP "<<time.toc()<<" ms\n";	
    PCL_INFO("num%d ICP精配准用时 %f ms\n",i+1,timecal.toc());
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    //showCloudsRight(points_with_normals_tgt, points_with_normals_src);
  }

  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  final_transform = targetToSource;
  std::cout << "\nICP 收敛, 拟合系数为 " << reg.getFitnessScore () << std::endl;
  //std::cout <<icp.getFinalTransformation ()<<std::endl;
  //print4x4Matrix(final_transform);

  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  //add the source to the transformed target
  //*output += *cloud_src;
  
  
 }


/* ---[ */
int main (int argc, char** argv)
{
  // Load data
  std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  loadData (argc, argv, data);

  // Check user input
  if (data.empty ())
  {
    PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]);
    PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc");
    return (-1);
  }
  PCL_INFO ("Loaded %d datasets.", (int)data.size ());
 

  PointCloud::Ptr result (new PointCloud), source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  PointCloud::Ptr temp (new PointCloud);
  PointCloud::Ptr finalcloud (new PointCloud);
  for (size_t i = 1; i < data.size (); ++i)
  {
    source = data[i-1].cloud;
    target = data[i].cloud;      
    *finalcloud=*source;
    PointCloud::Ptr init_result (new PointCloud);
    *init_result = *target;
    Eigen::Matrix4f init_transform = Eigen::Matrix4f::Identity ();
    PCL_INFO ("Aligning %s (%d) with %s (%d).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());
    sac_ia_align(source,target,init_result,init_transform,max_sacia_iterations,min_sample_dist,max_correspondence_dist);
    pairAlign (source, init_result, temp, pairTransform, true);
    pairTransform*=init_transform;   
    pcl::transformPointCloud (*temp, *result, GlobalTransform);
    *finalcloud+=*result;
    
    GlobalTransform = GlobalTransform * pairTransform;
    cout<<"粗+精变换:\n"<<GlobalTransform<<endl;
//显示
    viewPair(source,target,source,init_result);
    viewPair(source,target,source,result);
	
    //std::stringstream ss;
    //ss << i << ".pcd";
    //pcl::io::savePCDFile (ss.str (), *final, true);

  }
    
    return 0;

}
/* ]--- */
