import open3d as o3d
import numpy as np
import matplotlib as plt
import threading
import queue

class ViewerWithCallback:

    def __init__(self):

        self.flag_exit = False
        self.align_depth_to_color = True
        self.reg_p2p = None

        self.raw_intrinsic = [{
            "cx": 962.5819,
            "cy": 550.47235,
            "fx": 909.7707,
            "fy": 909.2633,
        }, {
            "cx": 962.88696,
            "cy": 549.53094,
            "fx": 911.055,
            "fy": 910.8141,
        }]

        self.translate_color_to_depth = [
            [-31.973124, -1.9932519, 3.9528272],
            [-31.951834, -2.1299322, 3.8983047]
        ]

# These intrinsics are pulled from my kinects' firmware.  They need to be changed, or queried, for other cameras.
        self.intrinsic_0 = o3d.camera.PinholeCameraIntrinsic(1920, 1080, self.raw_intrinsic[0]["fx"],
                                                             self.raw_intrinsic[0]["fy"],
                                                             self.raw_intrinsic[0]["cx"],
                                                             self.raw_intrinsic[0]["cy"])
        self.odo_intrinsic_0 = o3d.camera.PinholeCameraIntrinsic(self.intrinsic_0)
        self.intrinsic_1 = o3d.camera.PinholeCameraIntrinsic(1920, 1080, self.raw_intrinsic[1]["fx"],
                                                             self.raw_intrinsic[1]["fy"],
                                                             self.raw_intrinsic[1]["cx"],
                                                             self.raw_intrinsic[1]["cy"])

        # print(self.odo_intrinsic_0.intrinsic_matrix)

        config_0 = o3d.io.AzureKinectSensorConfig({
            "camera_fps": "K4A_FRAMES_PER_SECOND_15",
            "color_format": "K4A_IMAGE_FORMAT_COLOR_MJPG",
            "color_resolution": "K4A_COLOR_RESOLUTION_1080P",
            "depth_delay_off_color_usec": "0",
            "depth_mode": "K4A_DEPTH_MODE_NFOV_UNBINNED",
            "disable_streaming_indicator": "false",
            "subordinate_delay_off_master_usec": "0",
            "synchronized_images_only": "true",
            "wired_sync_mode": "K4A_WIRED_SYNC_MODE_MASTER"
        })

        config_1 = o3d.io.AzureKinectSensorConfig({
            "camera_fps": "K4A_FRAMES_PER_SECOND_15",
            "color_format": "K4A_IMAGE_FORMAT_COLOR_MJPG",
            "color_resolution": "K4A_COLOR_RESOLUTION_1080P",
            "depth_delay_off_color_usec": "0",
            "depth_mode": "K4A_DEPTH_MODE_NFOV_UNBINNED",
            "disable_streaming_indicator": "false",
            "subordinate_delay_off_master_usec": "160",
            "synchronized_images_only": "true",
            "wired_sync_mode": "K4A_WIRED_SYNC_MODE_SUBORDINATE"
        })

        self.sensors = [o3d.io.AzureKinectSensor(config_0), o3d.io.AzureKinectSensor(config_1)]
        self.sensors[0].connect(1)
        self.sensors[1].connect(0)

        self.rotation = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        self.transform = np.asarray([
            [0.6505933, -0.005149905, 0.7594089, -0.46503202], # -0.43403202
            [0.011280331, 0.9999322, -0.0028829677, -0.00505582],
            [-0.7593426, 0.010442023, 0.6506072, 0.190174], # 0.180174
            [0, 0, 0, 1]])

        self.frame_queue = queue.Queue(2)

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def update_transform(self, pcd_1, pcd_0):
        self.reg_p2p = o3d.pipelines.registration.registration_icp(pcd_1, pcd_0, 0.02, self.transform,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                  max_iteration=50))

        print(f'Transform After: {self.reg_p2p.transformation}')
        self.transform = self.reg_p2p.transformation

    def capture(self, cam_id_0, cam_id_1):
        while not self.flag_exit:
            rgbd_cap_0 = self.sensors[cam_id_0].capture_frame(self.align_depth_to_color)
            while rgbd_cap_0 is None:
                rgbd_cap_0 = self.sensors[cam_id_0].capture_frame(self.align_depth_to_color)
            c0, d0 = np.asarray(rgbd_cap_0.color).astype(np.uint8), np.asarray(rgbd_cap_0.depth).astype(np.float32)

            rgbd_cap_1 = self.sensors[cam_id_1].capture_frame(self.align_depth_to_color)
            while rgbd_cap_1 is None:
                rgbd_cap_1 = self.sensors[cam_id_1].capture_frame(self.align_depth_to_color)
            c1, d1 = np.asarray(rgbd_cap_1.color).astype(np.uint8), np.asarray(rgbd_cap_1.depth).astype(np.float32)

            depth_0 = o3d.geometry.Image(d0)
            img_0 = o3d.geometry.Image(c0)

            rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color=img_0,
                                                                        depth=depth_0,
                                                                        depth_scale=1000,
                                                                        depth_trunc=1.5,
                                                                        convert_rgb_to_intensity=False)

            depth_1 = o3d.geometry.Image(d1)
            img_1 = o3d.geometry.Image(c1)

            rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color=img_1,
                                                                        depth=depth_1,
                                                                        depth_scale=1000,
                                                                        depth_trunc=1.5,
                                                                        convert_rgb_to_intensity=False)

            self.frame_queue.put([rgbd_0, rgbd_1])

    def run(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)

        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        print("Sensor initialized. Press [ESC] to exit.")

        vis_geometry_added = False
        frame_count = 1
        pcd_0 = o3d.geometry.PointCloud()
        pcd_1 = o3d.geometry.PointCloud()

        translate = [-.0005, -.001, 0.008]

        self.reg_p2p = None

        evaluation = None

        transform_refinement_count = 3

        threading.Thread(target=self.capture, args=[0, 1]).start()

        while not self.flag_exit:
            rgbd_0, rgbd_1 = self.frame_queue.get()

            if not vis_geometry_added:
                pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_0,
                                                                       intrinsic=self.intrinsic_0)
                pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_1,
                                                                       intrinsic=self.intrinsic_1)

                vis.add_geometry(pcd_0)
                vis.add_geometry(pcd_1)
                vis_geometry_added = True
                continue

            pcd_0_new = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_0,
                                                                       intrinsic=self.intrinsic_0)

            # pcd_0_new = pcd_0_new.voxel_down_sample(voxel_size=0.005)
            # cl0, ind0 = pcd_0_new.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.0, print_progress=False)
            #
            # pcd_0_new = pcd_0_new.select_by_index(ind0)

            pcd_0.points = pcd_0_new.points
            pcd_0.colors = pcd_0_new.colors

            pcd_1_new = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_1,
                                                                       intrinsic=self.intrinsic_1)

            # pcd_1_new = pcd_1_new.voxel_down_sample(voxel_size=0.005)
            # cl, ind = pcd_1_new.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.0, print_progress=False)
            #
            # pcd_1_new = pcd_1_new.select_by_index(ind)

            pcd_1_new.transform(self.transform)

            pcd_1.points = pcd_1_new.points
            pcd_1.colors = pcd_1_new.colors

            if frame_count == 0:
                self.update_transform(pcd_1, pcd_0)

            if frame_count % 15 == 0:
                self.update_transform(pcd_1, pcd_0)
                transform_refinement_count -= 1
                evaluation = o3d.pipelines.registration.evaluate_registration(pcd_0, pcd_1, 0.02,
                                                                              self.reg_p2p.transformation)
                print(f'Registration Evaluation (AFTER P2P ICP): {evaluation}')



            # pcd_0.translate(translate)
            # pcd_1.translate(translate)

            # pcd_0.rotate(self.rotation)
            # pcd_1.rotate(self.rotation)

            vis.update_geometry(pcd_0)
            vis.update_geometry(pcd_1)

            vis.update_renderer()
            vis.poll_events()
            frame_count = frame_count + 1

        vis.destroy_window()
        self.sensors[0].disconnect()
        self.sensors[1].disconnect()


if __name__ == '__main__':
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Error) as cm:
        v = ViewerWithCallback()
        v.run()
