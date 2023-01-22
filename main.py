import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from lib.cam_calibration import Camcalib
from lib.img_operations import ImgOperator
from lib.lane_finder import LaneFinder, Line
from tqdm import tqdm


cam_calib = Camcalib()
img_operator = ImgOperator()
lanefinder = LaneFinder()


def image_runner(mtx, dst, img):
    undist_img = cam_calib.undistort(img, mtx, dst)
    undist_img = cv2.resize(
        undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA
    )
    rows, cols = undist_img.shape[:2]
    # undist_img = img_thresh.histogram_equalizer(undist_img)

    sobel_thresh = img_operator.gradient_combine_threshold(undist_img)
    hls_thresh = img_operator.hls_combined_threshold(undist_img)
    combined_thresh = img_operator.combined_hls_sobel_threshold(
        sobel_thresh, hls_thresh
    )

    img_size = combined_thresh.shape
    s_ltop, s_rtop = [img_size[1] / 2 - 24, 5], [img_size[1] / 2 + 24, 5]
    s_lbottom, s_rbottom = [110, img_size[0]], [img_size[1] - 50, img_size[0]]
    src = np.float32([s_lbottom, s_ltop, s_rtop, s_rbottom])
    dst = np.float32([(150, 720), (150, 0), (500, 0), (500, 720)])

    # Get warped image and Warp transformation
    warped_img, M, Minv = img_operator.warp_image(combined_thresh, src, dst, (720, 720))
    # Initiate Lines
    left_line = Line()
    right_line = Line()
    lane_img = lanefinder.blind_search(warped_img, left_line, right_line)

    w_comb_result, w_color_result = lanefinder.draw_lane(
        lane_img, left_line, right_line
    )

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (img_size[1], img_size[0]))
    comb_result = np.zeros_like(undist_img)
    comb_result[img_operator.hcr : rows - 10, 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, comb_result, 0.3, 0)
    lanefinder.draw_road_info(result, left_line, right_line)

    return result


def runOnVideo(video, maxFrames, mtx, dst):
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("Please check the Video path and Video is corrupted or not")
            break
        visualization = image_runner(mtx, dst, frame)

        yield visualization

        readFrames += 1

        if readFrames > maxFrames:
            print("Requested more frames read than existing frames")
            break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Path to Input")
    parser.add_argument(
        "-it",
        "--input_type",
        required=False,
        default="image",
        choices=["image", "video"],
        help="Input type => Image or Video ?",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        default="./output_images/",
        help="Output Path - Folder Path only",
    )

    args = parser.parse_args()

    mtx, dst = cam_calib.calib()

    if args.input_type == "video":
        output_path = (
            args.output_path
            + "/"
            + args.input.split("/")[-1].split(".")[0]
            + "_output.mp4"
        )
        cap = cv2.VideoCapture(args.input)
        _, test_img = cap.read()
        test_result = image_runner(mtx, dst, test_img)
        cam_calib.calculate_projection_err()
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_width = test_result.shape[0]
        output_height = test_result.shape[1]
        cap.release()
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps=frames_per_second,
            frameSize=(output_height, output_width),
        )
        video = cv2.VideoCapture(args.input)

        for visualization in tqdm(
            runOnVideo(video, num_frames, mtx, dst), total=num_frames
        ):
            # cv2.imshow("Output", visualization)  # Visualize
            # cv2.waitKey(50)  # 50 millisec
            video_writer.write(visualization)  # Write to a Video
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()

    else:
        output_path = (
            args.output_path + args.input.split("/")[-1].split(".")[0] + "_output.jpg"
        )
        print(output_path)
        cam_calib.calculate_projection_err()
        img = cv2.imread(args.input)
        result = image_runner(mtx, dst, img)
        cv2.imwrite(output_path, result)


if __name__ == "__main__":
    main()
