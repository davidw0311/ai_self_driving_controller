source ~/ros_ws/devel/setup.bash
# gnome-terminal -e  
cd ~/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts
./run_sim.sh
roslaunch my_controller process_image.launch
