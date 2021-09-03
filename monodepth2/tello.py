from djitellopy import Tello
import cv2
import os
import matplotlib as mpl
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from torch._C import parse_schema
import networks
import torch
from torchvision import transforms
import time


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


model_name = "mono_640x192"

encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k,
                     v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()

tello = Tello()
tello.connect()

tello.streamon()

# cam = cv2.VideoCapture(0)

tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)

while True:
    input_image = tello.get_frame_read().frame  # cam.read()  ret,
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    original_width, original_height = (640, 192)  # input_image.shape
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = cv2.resize(
        input_image, (feed_width, feed_height), interpolation=cv2.INTER_AREA)  # .astype(np.float32) / 255.

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    # vmax = np.percentile(disp_resized_np, 95)

    # plt.figure(figsize=(10, 10))
    # plt.subplot(211)
    # plt.imshow(input_image)
    # plt.title("Input", fontsize=22)
    # plt.axis('off')

    # plt.subplot(212)
    # plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    # plt.title("Disparity prediction", fontsize=22)
    # plt.axis('off')

    # Bereich zum suchen
    small_fov_out = disp_resized_np[48:96, 160:320]

    # Nächste punkt
    max = np.max(small_fov_out)
    position_min = np.where(disp_resized_np == max)

    # Einzeichnen
    # cv2.putText(disp_resized_np, "min: " + str(min))
    image = cv2.circle(disp_resized_np, (position_min[1][0]+5, position_min[0][0]+5), radius=10,
                       color=(0, 0, 255), thickness=-1)
    print(max)

    if(max > 0.4):

        step_size = 25
        size = 50
        passed_cubes = []
        for y in range(0, original_height - size - 1, step_size):
            for x in range(0, original_width - size - 1, step_size):
                # image = cv2.rectangle(disp_resized_np, (x, y), (x + size, y + size ), (255, 0, 0), 2)
                part_img = disp_resized_np[y:y + size, x:x + size]
                for array in part_img:
                    for num in array:
                        if num > 0.2:
                            viabile_img_part = False
                            break
                        else:
                            viabile_img_part = True
                    if(viabile_img_part == True):
                        # image = cv2.rectangle(
                        #     disp_resized_np, (x, y), (x + size, y + size), (255, 0, 0), 2)
                        passed_cubes.append((x + (size / 2), y + (size / 2)))
        if(len(passed_cubes) == 0):
            tello.send_rc_control(0, 0, 30, 0)

            # print("land")
            # tello.land
            # break
        else:
            middleX = original_width / 2
            middleY = original_height / 2

            differences = []

            for cube in passed_cubes:
                total_difference = 0
                if(cube[0] > middleX and cube[1] > middleY):
                    total_difference = (
                        cube[0] - middleX) + (cube[1] + middleY)
                if(cube[0] < middleX and cube[1] > middleY):
                    total_difference = (
                        middleX - cube[0]) + (cube[1] + middleY)
                if(cube[0] < middleX and cube[1] < middleY):
                    total_difference = (
                        middleX - cube[0]) + (middleY + cube[1])
                if(cube[0] > middleX and cube[1] < middleY):
                    total_difference = (
                        cube[0] - middleX) + (middleY + cube[1])
                differences.append(total_difference)

            min_varieation_idx = np.min(differences)

            # least_variation = find_nearest(passed_cubes, (middleX, middleY))
            least_variation = passed_cubes[min_varieation_idx]

            image = cv2.rectangle(
                disp_resized_np,
                ((int(least_variation[0])) - int((size / 2)),
                 int(least_variation[1]) - int((size / 2))),
                ((int(least_variation[0])) + int((size / 2)),
                 int(least_variation[1]) + int((size / 2))),
                (255, 0, 0),
                2
            )

            print(least_variation)

            tello.send_rc_control(0, 0, 0, 0)

            time.sleep(3)

            tello.land()

            # for()
        # tello.send_rc_control(0, -20, 0, 0)
        # print("land")
        # tello.land
        # break
    else:
        tello.send_rc_control(0, 30, 0, 0)

    cv2.imshow("test", disp_resized_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
