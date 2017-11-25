#include <iostream>
#include <string>
#include <pcl/console/time.h>   // TicToc
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event,
                       void* nothing)
{
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration = true;
}

int main (int argc, char* argv[])
{
  PointCloudT::Ptr cloud_in (new PointCloudT);  // Original point cloud
  PointCloudT::Ptr cloud_tr (new PointCloudT);  // Transformed point cloud
  PointCloudT::Ptr cloud_icp (new PointCloudT);  // ICP output point cloud
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
 
  if (argc < 2)
  {
    printf ("Usage :\n");
    printf ("\t\t%s source.pcd  target.pcd  number_of_ICP_iterations\n", argv[0]);
    PCL_ERROR ("Provide two pcd file.\n");
    return (-1);
  }
  int iterations = 1000;  // Default number of ICP iterations
  if (argc > 3)
  {
    // If the user passed the number of iteration as an argument
    iterations = atoi (argv[3]);
    if (iterations < 1)
    {
      PCL_ERROR ("Number of initial iterations must be >= 1\n");
      return (-1);
    }
  }

  pcl::console::TicToc time;
  time.tic ();

  if (pcl::io::loadPCDFile (argv[1], *cloud_icp) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
    return (-1);
  }
  std::cout << "\nLoaded file " << argv[1] << " (" << cloud_icp->size () << " points) in " << time.toc () << " ms\n" << std::endl;

  time.tic ();
  if (pcl::io::loadPCDFile (argv[2], *cloud_in) < 0)
  {
    PCL_ERROR ("Error loading cloud %s.\n", argv[2]);
    return (-1);
  }
  std::cout << "\nLoaded file " << argv[2] << " (" << cloud_in->size () << " points) in " << time.toc () << " ms\n" << std::endl;

  pcl::VoxelGrid<PointT> voxel;
  if(cloud_in->size()>10000)
  {
    
    voxel.setInputCloud (cloud_in);
    voxel.setLeafSize (0.001f, 0.001f, 0.001f);
    voxel.filter (*cloud_in);

    std::cout << "PointCloud after filtering: " << argv[2] << " (" << cloud_in->size () <<") "<< std::endl;
    
  }
  if(cloud_icp->size()>10000)
  {
    
    voxel.setInputCloud (cloud_icp);
    voxel.setLeafSize (0.001f, 0.001f, 0.001f);
    voxel.filter (*cloud_icp);

    std::cout << "PointCloud after filtering: " << argv[1] << " (" << cloud_icp->size () <<") "<< std::endl;
    
  }

  *cloud_tr = *cloud_icp;  // We backup cloud_icp into cloud_tr for later use

  // The Iterative Closest Point algorithm
  time.tic ();
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setMaximumIterations (iterations);
  icp.setInputSource (cloud_icp);
  icp.setInputTarget (cloud_in);
  icp.align (*cloud_icp);
  icp.setMaximumIterations (30);  // We set this variable to 1 for the next time we will call .align () function
  std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc () << " ms" << std::endl;

  if (icp.hasConverged ())
  {
    std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
    std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
    transformation_matrix = icp.getFinalTransformation ().cast<double>();       
    std::cout <<icp.getFinalTransformation ()<<std::endl;
  }
  else
  {
    PCL_ERROR ("\nICP has not converged.\n");
    return (-1);
  }
 
  pcl::visualization::PCLVisualizer viewer ("ICP demo");

  int v1 (0);
  int v2 (1);
  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

  float bckgr_gray_level = 0.0;  // Black
  float txt_gray_lvl = 1.0 - bckgr_gray_level;

  // Original point cloud is white
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h (cloud_in, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                             (int) 255 * txt_gray_lvl);
  viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
  viewer.addPointCloud (cloud_in, cloud_in_color_h, "cloud_in_v2", v2);

  // Transformed point cloud is green
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h (cloud_tr, 20, 180, 20);
  viewer.addPointCloud (cloud_tr, cloud_tr_color_h, "cloud_tr_v1", v1);

  // ICP aligned point cloud is red
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h (cloud_icp, 20, 180, 20);
  viewer.addPointCloud (cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

  viewer.addText ("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
  viewer.addText ("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

  std::stringstream ss;
  ss << iterations;
  std::string iterations_cnt = "ICP iterations = " + ss.str ();
  viewer.addText (iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);
  
  viewer.setSize (1280, 1024);  // Visualiser window size
    
  viewer.registerKeyboardCallback (&keyboardEventOccurred, (void*) NULL);

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
 
    if (next_iteration)
    {   
      time.tic ();
      icp.align (*cloud_icp);
      std::cout << "Applied 1 ICP iteration in " << time.toc () << " ms" << std::endl;

      if (icp.hasConverged ())
      {
        printf ("\033[11A");  // Go up 11 lines in terminal output.
        printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore ());
        std::cout << "\nICP transformation " << ++iterations << " : cloud_icp -> cloud_in" << std::endl;
        transformation_matrix *= icp.getFinalTransformation ().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
        std::cout<<transformation_matrix<<endl;  // Print the transformation between original pose and current pose

        ss.str ("");
        ss << iterations;
        std::string iterations_cnt = "ICP iterations = " + ss.str ();
        viewer.updateText (iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
        viewer.updatePointCloud (cloud_icp, cloud_icp_color_h, "cloud_icp_v2");
      }
      else
      {
        PCL_ERROR ("\nICP has not converged.\n");
        return (-1);
      }
    }
    next_iteration = false;
  }
  return (0);
}