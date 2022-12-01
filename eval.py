from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import cv2

from array import *

#########midas import

import os
import glob
import torch
import utils
import cv2
import argparse
import numpy as np#ssssssssssssssssssssssssssssssssssssssssssssssssssss

from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image
import matplotlib.pyplot as plt#ssssssssssssssssssssssssssssssssssssssssssssssssssss

from os import environ#ssssssssssssssssssssssssssssssssssssssssssssssssssss
############################ Main Integration CODE ############################
import speech_recognition as sr
import time
from gtts import gTTS

############################ Main Integration CODE ############################

coco_class_dic_with_index = {
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorbike': 4,
    'aeroplane': 5,
    'bus': 6,
    'train': 7,
    'truck': 8,
    'boat': 9,
    'bench': 14,
    'bird': 15,
    'cat': 16,
    'dog': 17,
    'horse': 18,
    'sheep': 19,
    'cow': 20,
    'elephant': 21,
    'zebra': 23,
    'giraffe': 24,
    'backpack': 25,
    'umbrella': 26,
    'handbag': 27,
    'tie': 28,
    'suitcase': 29,
    'frisbee': 30,
    'skis': 31,
    'snowboard': 32,
    'kite': 34,
    'skateboard': 37,
    'surfboard': 38,
    'bottle': 40,
    'cup': 42,
    'fork': 43,
    'knife': 44,
    'spoon': 45,
    'bowl': 46,
    'banana': 47,
    'apple': 48,
    'sandwich': 49,
    'orange': 50,
    'broccoli': 51,
    'carrot': 52,
    'pizza': 54,
    'doughnut': 55,
    'cake': 56,
    'chair': 57,
    'sofa': 58,
    'potted plant': 59,
    'bed': 60,
    'dining table': 61,
    'toilet': 62,
    'laptop': 64,
    'mouse': 65,
    'remote': 66,
    'keyboard': 67,
    'microwave': 69,
    'oven': 70,
    'toaster': 71,
    'sink': 72,
    'refrigerator': 73,
    'book': 74,
    'clock': 75,
    'vase': 76,
    'scissors': 77,
    'toothbrush': 80,
    'teddy bear': 78,
    'hair drier': 79,
    'tv monitor': 63,
    'cell phone': 68,
    'hot dog': 53,
    'wine glass': 41,
    'tennis racket': 39,
    'baseball bat': 35,
    'baseball glove': 36,
    'sports ball': 33,
    'traffic light': 10,
    'fire hydrant': 11,
    'stop sign': 12,
    'parking meter': 13
}
coco_class_dic_withspaces = {
    'teddy': -1, 'bear': 78,
    'hair': -1, 'drier': 79,
    'tv': -1, 'monitor': 63,
    'mobile':-1,
    'cell': -1, 'phone': 68,
    'hot': -1, 'dog': 53,
    'wine': -1, 'glass': 41,
    'tennis': -1, 'racket': 39,
    'baseball': -1, 'bat': 35, 'glove': 36,
    'sports': -1, 'ball': 33,
    'traffic': -1, 'light': 10,
    'fire': -1, 'hydrant': 11,
    'stop': -1, 'sign': 12,
    'parking': -1, 'meter': 13,
    'dining': -1, 'table': 61,
    'potted': -1, 'plant': 59
}


def get_all_coco_class_names(input):  # input is a list#this is the funtion needed to extract in main project
    #converting upper to lower case, fix for the error of not matching in the dictnary:
    # splitting by individual objects :
    # this will store all objects individualy :
    total_raw_input = (input.lower()).split()  # total_raw_input is a list

    print('FROM get_all_coco_class_names :-> \n the given input was : ' + str(total_raw_input))
    # new object holder :
    objects = {'obj1': 0}
    # objects = ['obj1']
    objects.pop('obj1')  # poping a temp variable
    # now we need to check if these objects are in the coco class :
    for index, object in enumerate(total_raw_input):
        # check to see if the word is in the diconary with spaces :
        if coco_class_dic_withspaces.get(object, 0) < 0:
            # it is one of the words with spaces : checking the index right after the current one
            # need to cater for out of bounds in this segment :
            if coco_class_dic_withspaces.get(total_raw_input[index + 1], 0) > 0:
                # the next word is in the dic so we need to :
                # pop both words and concatinate as one with spaces :
                holder = str(object) + " " + str(total_raw_input[index + 1])

                # print(holder)
                # add it to the list of words :
                objects[holder] = coco_class_dic_withspaces.get(total_raw_input[index + 1], 0)
                # objects.append(holder)
        # checking the word in the regular dic :
        elif coco_class_dic_with_index.get(object, 0) > 0:
            # the object was found in class
            # if the objects class name is with spacing then pop the firs one and add it in the next index :
            # print(object)

            objects[object] = coco_class_dic_with_index.get(object, 0)
            # objects.append(object)  # saving the word found in the list in the finial list.
        # ele s  leave keep the loop going
    return objects
def RecognizeSpeechAndReturnObjectsAndIndexs():#this is the funtion needed to extract in main project#  is the object there or not (if the object is not there so I need to recheck the 'find' part
    # making instances of the reconigser class:
    r2 = sr.Recognizer()  # to get the objects names
    #r3 = sr.Recognizer()  # to get first input
    # telling where the input is conming from :
    with sr.Microphone() as source:
       # print('say keyword "Find" then say the object to locate')
        time.sleep(1)  # wait 1 sec to let the user read and react.
        # adjusting the to the current ambient noise of the room :
        while True:
            with sr.Microphone() as source:
                r3 = sr.Recognizer()  # to get first input
                r3.adjust_for_ambient_noise(source, duration=0.5)
                print('say keyword "Find" then say the object to locate')
                print('speak now')
                audio = r3.listen(source,timeout=5,phrase_time_limit=5)  # r3 will listen from the microphone and then store it in audio
                try:
                    if 'find' in r3.recognize_google(
                            audio):  # r2 will comvert to text to speach and then the if condition will  see if the text is the keyword we choose
                        # the keyword has been found and hence we will work accordingly :
                        r2 = sr.Recognizer()
                        print("keyword has been said")
                        # getting an other input from the user after using the keyword :
                        with sr.Microphone() as source:
                            print('keyword found say object : ')
                            audio_objects = r2.listen(source,timeout=5,phrase_time_limit=5)

                            try:
                                output = r2.recognize_google(audio_objects)
                                print(output)
                                # the function below will take the input and then return a list with all the words in it :
                                res = get_all_coco_class_names(output.__str__())  # output.str is a string
                                #TextToSpeech(output)
                               # print('these are the objects we can find for you : ' + output.__str__()(res.keys()) + ' indexs for the keys :' + output.__str__()(res.values()))
                                #findObjects(output,)
                                return list(res.values())

                                # working to make object the class to look for
                            # excepts for errors

                            except sr.RequestError as e:
                                print('Request error \n' + 'failed'.format(e))
                            except sr.UnknownValueError:
                                print("unknown value error")
                                str = "Your desired object could Not Found"
                                TextToSpeech(str)
                                # moeez's errors loop needs to break here and it does not.
                                del r3
                                # returrns to the top of the loop
                    else:
                        str = "Your said the wrong keyword"
                        TextToSpeech(str)
                        # print('Not Found')
                       # print("could not find")
                except:
                    print("keyword haven't been spoken")

def TextToSpeech(results):#this is the funtion needed to extract in main project
    # text to speech
    myText = results
    language = 'en'
    Output = gTTS(text=myText, lang=language, slow=False)
    Output.save("audio.mp3")
    os.system("start audio.mp3")
    return results




############################ MIDAS CODE ############################


def suppress_qt_warnings():#ssssssssssssssssssssssssssssssssssssssssssssssssssss
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def MidasImageDistanceEvaluation(img,transform,optimize,device,model):
    print("start processing")
    # ---------------------------------------------------------------------------------------------------------
    # for ind, img_name in enumerate(img_names):
    # print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
    # input
    # cv2.imshow("image in midas:",img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    ##ssssssssssssssssssssssssssssssssssssssssssssssssssss

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )

    img_depth = MyDepthCalculation(prediction,2)        # output#ssssssssssssssssssssssssssssssssssssssssssssssssssss
    cv2.imshow("midas img_depth",img_depth)
    return img_depth
    print("Depth image calculated finished-MidasImageDistanceEvaluation")
    # ---------------------------------------------------------------------------------------------------------
    # URL = "http://192.168.0.103:8080/video"
    # cam = cv2.VideoCapture(URL)
    # while True:
    #     check, image = cam.read()
    #     cv2.imshow('IPWebcam', image)
    #     # height, width, channels = img.shape
    #     # Nawfal
    #     if image.ndim == 2:
    #         img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #
    #     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    #     # ---------------------------------------------------------------------------------------------------------
    #     img_input = transform({"image": img})["image"]
    #
    #     # compute
    #     with torch.no_grad():
    #         sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
    #         if optimize == True and device == torch.device("cuda"):
    #             sample = sample.to(memory_format=torch.channels_last)
    #             sample = sample.half()
    #         prediction = model.forward(sample)
    #         prediction = (
    #             torch.nn.functional.interpolate(
    #                 prediction.unsqueeze(1),
    #                 size=img.shape[:2],
    #                 mode="bicubic",
    #                 align_corners=False,
    #             )
    #                 .squeeze()
    #                 .cpu()
    #                 .numpy()
    #         )
    #     # output
    #     # filename = os.path.join(
    #     #     output_path, os.path.splitext(os.path.basename(img_name))[0]
    #     # )
    #
    #     # cv2.imshow("filename", prediction)
    #     # time.sleep(10)
    #     print("prediction:", prediction)
    #     # utils.write_depth(filename, prediction, bits=2)
    #     MyDepthCalculation(prediction, 2)  # output
    #     # blur = cv2.GaussianBlur(prediction, (5, 5), 0)
    #     # cv2.imshow('image', blur)
    #     # cv2.waitKey(10)
    #     if cv2.waitKey(1) == 27:
    #         break


def runn(model_path, model_type="large", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "large":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
    elif model_type == "small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize==True:
        rand_example = torch.rand(1, 3, net_h, net_w)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)
    print("Midas init finished-runn")
    return transform,optimize,device,model
    #
    # #the input part
    # img_names = glob.glob(os.path.join(input_path, "*"))
    # num_images = len(img_names)
    #
    # # create output folder
    # os.makedirs(output_path, exist_ok=True)
    #
    # print("start processing")
    # # ---------------------------------------------------------------------------------------------------------
    #
    # URL = "http://192.168.0.103:8080/video"
    # cam = cv2.VideoCapture(URL)
    # while True:
    #     check, image = cam.read()
    #     cv2.imshow('IPWebcam', image)
    #     # height, width, channels = img.shape
    #     # Nawfal
    #     if image.ndim == 2:
    #         img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #
    #     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    #     # ---------------------------------------------------------------------------------------------------------
    #     img_input = transform({"image": img})["image"]
    #
    #     # compute
    #     with torch.no_grad():
    #         sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
    #         if optimize == True and device == torch.device("cuda"):
    #             sample = sample.to(memory_format=torch.channels_last)
    #             sample = sample.half()
    #         prediction = model.forward(sample)
    #         prediction = (
    #             torch.nn.functional.interpolate(
    #                 prediction.unsqueeze(1),
    #                 size=img.shape[:2],
    #                 mode="bicubic",
    #                 align_corners=False,
    #             )
    #                 .squeeze()
    #                 .cpu()
    #                 .numpy()
    #         )
    #     # output
    #     # filename = os.path.join(
    #     #     output_path, os.path.splitext(os.path.basename(img_name))[0]
    #     # )
    #
    #     # cv2.imshow("filename", prediction)
    #     # time.sleep(10)
    #     print("prediction:", prediction)
    #     # utils.write_depth(filename, prediction, bits=2)
    #     MyDepthCalculation(prediction, 2)  # output
    #     # blur = cv2.GaussianBlur(prediction, (5, 5), 0)
    #     # cv2.imshow('image', blur)
    #     # cv2.waitKey(10)
    #     if cv2.waitKey(1) == 27:
    #         break
    # print("finished")

# ---------------------------------------------------------------------------------------------------------
def MyDepthCalculation(depth,bits):#ssssssssssssssssssssssssssssssssssssssssssssssssssss
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
     # cv2.imshow("depthPic",out.astype("uint16"))
    img_depth=out.astype("uint16")
    print("depth image made -MyDepthCalculation")
    return img_depth

    # print("img",type(img_depth),len(img_depth),img_depth)
    # print("img",img_depth.shape)
    # print("img",type(img),"len 1:",len(img[0]),"len2 ",len(img),img)
    # read files and put in array and take average
    # fBottle = open("bottle.txt", "r")
    # # fCup = open("cup.txt", "r")
    # # fMouse = open("mouse.txt", "r")
    # fCup = open("cup.txt", "r")
    # fMouse = open("mouse.txt", "r")
    #testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # midasOutput file name/same name directory in yoloact


    # num_dets_to_consider=3
    # directory="D:/FYP/Phase4/integration/yolact-master/midasOutput"
    # yoloactOutput=open(directory + "/midasOutput.txt", "r")
    # yoloactOutput_array= yoloactOutput.read().split("\n")
    # print("well yoloact loop is:",yoloactOutput_array)
    # coordinatesOfObjects = [[] for j in range(num_dets_to_consider)]
    # #Generic testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # for i in range(int(yoloactOutput_array[0])):
    #     yoloactCoordinates = open(yoloactOutput_array[i+1]+".txt", "r")
    #     yoloactTempCoordinates = yoloactCoordinates.readline().split(" ")
    #     for x in yoloactTempCoordinates:
    #         if x is '':
    #             print("done")
    #         else:
    #             tempList = x.split(",")
    #             coordinatesOfObjects[i].append([int(tempList[0]), int(tempList[1])])

    #Generic testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    # fBottle = open("bottle.txt", "r")
    # arr = [[] for j in range(num_dets_to_consider)]
    # # coordinatesOfObjects = [[]] * num_dets_to_consider
    # coordinatesOfObjects = [[] for j in range(num_dets_to_consider)]
    # botTempArr = fBottle.readline().split(" ")
    # print("this is length of bottle array", len(botTempArr))
    # for x in botTempArr:
    #     if x is '':
    #         print("done")
    #     else:
    #         tempList = x.split(",")
    #         coordinatesOfObjects[0].append([int(tempList[0]), int(tempList[1])])
    # #testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fCup = open("cup.txt", "r")
    # cupTempArr = fCup.readline().split(" ")
    # print("this is length of bottle array", len(botTempArr))
    # for x in cupTempArr:
    #     if x is '':
    #         print("done")
    #     else:
    #         tempList = x.split(",")
    #         coordinatesOfObjects[1].append([int(tempList[0]), int(tempList[1])])
    #
    # #testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fMouse = open("mouse.txt", "r")
    # cupTempArr = fMouse.readline().split(" ")
    # print("this is length of bottle array", len(botTempArr))
    # for x in cupTempArr:
    #     if x is '':
    #         print("done")
    #     else:
    #         tempList = x.split(",")
    #         coordinatesOfObjects[2].append([int(tempList[0]), int(tempList[1])])
    # #testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #average +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # number_of_avg =0
    # avg_sum= np.uint64(0)
    # avg_SeqWise=[]
    # for j in range(num_dets_to_consider):
    #     for pair in  coordinatesOfObjects[j]:
    #         avg_sum+=img_depth[pair[1]][pair[0]]
    #         number_of_avg+=1
    #     avg_SeqWise.append(avg_sum/number_of_avg)
    # for j in range(num_dets_to_consider):
    #     print("the avg seq wise is :",avg_SeqWise[j])
    #average +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # coordinatesOfObjects = [[]] * num_dets_to_consider

    # botTempArr=fBottle.readline().split(" ")
    # print("this is length of bottle array",len(botTempArr))
    # for x in botTempArr:
    #     tempList=x.split(",")
    #     coordinatesOfObjects[0].append([int(tempList[0]),int(tempList[2])])
    # for x in coordinatesOfObjects[0]:
    #     print(x)
    # coordinatesOfObjects[j].append([col,row])#x and y
    # plt.imshow(img)pixel_values
    # cv2.imshow("img cv2",img)#ssssssssssssssssssssssssssssssssssssssssssssssssssss
    # cv2.waitKey(10000)  # ssssssssssssssssssssssssssssssssssssssssssssssssssss


# ---------------------------------------------------------------------------------------------------------
def AverageObjectDistanceCalculation(num_dets_to_consider,coordinatesOfObjects,img_depth):
    number_of_avg = 0
    avg_sum = np.uint64(0)
    avg_SeqWise = []
    for j in range(num_dets_to_consider):
        for pair in coordinatesOfObjects[j]:
            avg_sum += img_depth[pair[1]][pair[0]]
            number_of_avg += 1
        avg_SeqWise.append(avg_sum / number_of_avg)
    # for j in range(num_dets_to_consider):
    #     print("the avg seq wise is :", avg_SeqWise[j])#-----------------avg_SeqWise[j]-------
    print("Avg calculated-AverageObjectDistanceCalculation")
    return avg_SeqWise
# ---------------------------------------------------------------------------------------------------------

def MidasnetINIT():
    suppress_qt_warnings()#ssssssssssssssssssssssssssssssssssssssssssssssssssss
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    transform,optimize,device,model=runn("model-f6b98070.pt")
    print("midas init- father:runn -MidasnetINIT")
    return transform,optimize,device,model
    # compute depth maps
    #runn(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize)
    # runn("input_output/input", "input_output/output", "weights/yolact_base_54_800000.pth", "large", True)


############################ MIDAS CODE ############################




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})




def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=1, fps_str=''):#sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save
#lkllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            all_objects_mask = t[3][:args.top_k]
            #my code:
            # for y in all_objects_mask:
            #     for x in y:
            #         if(all_objects_mask[y][x]==true)
            #my code:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        # print(type(tf.print(all_objects_mask)))
        # print("all_objects_mask",all_objects_mask.print())
        # with all_objects_mask.Session() as sess:  print(all_objects_mask.eval())

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            # rgb_Color=COLORS[color_idx]
            return color_cache[on_gpu][color_idx],COLORS[color_idx]
        else:
            color = COLORS[color_idx]
            rgb_Color=color
            # print("before this may be the color:", color,"classes:",cfg.dataset.class_names[classes[j]])
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])

                # print("this may be the color:",color)
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
                # print("Cache:",color_cache," color:",color)

            # print("my colors", color, "class:", cfg.dataset.class_names[classes[j]])

        return color,rgb_Color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # print("masks",masks.view)
        RGB_list=[]
        object_class_index=[]
        all_Bounding_Box=[]
        for j in range(num_dets_to_consider):
            color,temp_RGB_arr_classes=get_color(j, on_gpu=img_gpu.device.index)
            RGB_list.append(temp_RGB_arr_classes)
            object_class_index.append(classes[j])
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colorList=[]
        for j in range(num_dets_to_consider):
            tempcolor,temprgb=get_color(j, on_gpu=img_gpu.device.index)
            colorList.append(tempcolor.view(1, 1, 1, 3))
        colors = torch.cat(colorList, dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        # print("colors:",colors,"\n\n")
        # print("masks_color:",masks_color.shape,"\n\n")

        # start done for conversion of tensor to picture
        # from torchvision import transforms
        # tensor_to_pil = transforms.ToPILImage()(masks_color.squeeze_(0))
        #/ ending done for conversion of tensor to picture


        # x = torch.FloatTensor([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]])
        # converted_tensor = torch.nn.functional.embedding(masks_color, x).permute(0, 3, 1, 2)
        # print("convertedTensor:",converted_tensor)
        # cv2.imshow("converted_tensor",converted_tensor)
        # cv2.waitKey(1000)
        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # print("iv masks",inv_alph_masks.shape)

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        # print("inv_alph_masks",inv_alph_masks,"masks_color_summand",masks_color_summand,"masks_color",masks_color)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    # cv2.imshow("img_numpys", img_numpy)
    # cv2.waitKey(1000)
    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        object_class_index=[]
        coordinatesOfObjects=[]
        all_Bounding_Box=[]
        return num_dets_to_consider,object_class_index,coordinatesOfObjects,all_Bounding_Box#if there is nothing detected
    # sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss#sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss(only commented)
    if args.display_text or args.display_bboxes:
        for j in range(num_dets_to_consider):#its nawfal, I donr know much lekin im removing the reversed thingy #for j in reversed(range(num_dets_to_consider))
            x1, y1, x2, y2 = boxes[j, :]
            # print("num_dets_to_consider",num_dets_to_consider,"number",range(num_dets_to_consider))
            all_Bounding_Box.append(((x1, y1), (x2, y2)))

            # color = get_color(j)
            # score = scores[j]
            color=(0, 0, 255)
            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2),color,1)
                pic=img_numpy[y1:y2,x1:x2]
                # cv2.imshow("img", img_numpy)
                # cv2.imshow("mypic",pic)
                # cv2.waitKey(10000)


            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                # print(_class)
        for j in range(num_dets_to_consider):
            cv2.rectangle(img_numpy, all_Bounding_Box[j][0], all_Bounding_Box[j][1], (255,255,255) , 1)
            # print("object_class_index", object_class_index[j],"class names:",cfg.dataset.class_names[object_class_index[j]], "RGB_list",RGB_list[j], "rectangle  (x1, y1), (x2, y2)",
            #       all_Bounding_Box[j], " j:", j)
        cv2.imshow("After my bounding box:", img_numpy)
        coordinatesOfObjects=[[] for j in range(num_dets_to_consider)]

        # cfg.dataset.class_names[object_c
        # for fileInput in range(num_dets_to_consider):
        # print("middle value:",img_numpy[all_Bounding_Box[j][0][1]+int(all_Bounding_Box[j][1][1]/2)][all_Bounding_Box[j][0][0]+int(all_Bounding_Box[j][1][0]/2)])
        for j in range(num_dets_to_consider):
            r, g, b = 0, 1, 2
            # row,col=0,0
            if object_class_index[j] == 61:#dining table so ignore it
                continue
            else:
                for row in range(all_Bounding_Box[j][0][1], (all_Bounding_Box[j][1][1])):
                    for col in range(all_Bounding_Box[j][0][0], (all_Bounding_Box[j][1][0])):
                        if not(row>=h-1 or col>= w-1):
                            if img_numpy[row][col][b] == RGB_list[j][r] and img_numpy[row][col][g] == RGB_list[j][g] and img_numpy[row][col][r] == RGB_list[j][b]:#image is bgr
                                coordinatesOfObjects[j].append([col,row])#x and y
            print("objects found:"+cfg.dataset.class_names[object_class_index[j]])
            if object_class_index[j] == 61:#dining table so ignore it it gows aik peeche shift+tab
                num_dets_to_consider-=1
                object_class_index.remove(60)

            # print("my image size",img_numpy.shape)
            # print("my array size",len(arr))
            # for row in range(all_Bounding_Box[j][0][1], (all_Bounding_Box[j][1][1])):
            #     for col in range(all_Bounding_Box[j][0][0], (all_Bounding_Box[j][1][0])):
            # for i in range(len(coordinatesOfObjects[j])):
            #     print(coordinatesOfObjects[j][i],end=",")
            # # print(coordinatesOfObjects[j])
            # print("\nrange(coordinatesOfObjects[j])",coordinatesOfObjects[j][0][1])
            # print(len(arr))
            # print(len(arr[0]))
            # print(arr[0])
            # print(arr[0][0])
            # arr[0][0]=1
            # print(arr[0][0])
            # print(arr[1][0])

            # for i in range(int(len(coordinatesOfObjects[j])/2)):
            #         arr[coordinatesOfObjects[j][i][1]][coordinatesOfObjects[j][i][0]]=1
            #         # print("y:"+str(coordinatesOfObjects[j][i][1]),",x:"+str(coordinatesOfObjects[j][i][0]))
            # for r in range(len(arr)):
            #     for c in range(len(arr[r])):
            #         # print("(",r,c,")", end="")
            #         print(arr[r][c], end="")
            #     print("")
            # plt.imshow(coordinatesOfObjects[j])
            # plt.show()
            # cv2.imshow("img", img_numpy)
    #         if(_class == "bottle"):
    #             print("equal")

    # sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss#sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss(only commented)
    # if args.display_text or args.display_bboxes:
    #     for j in reversed(range(num_dets_to_consider)):
    #         x1, y1, x2, y2 = boxes[j, :]
    #         color = get_color(j)
    #         score = scores[j]
    #
    #         if args.display_bboxes:
    #             cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
    #
    #         if args.display_text:
    #             _class = cfg.dataset.class_names[classes[j]]
    #             text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
    #
    #             font_face = cv2.FONT_HERSHEY_DUPLEX
    #             font_scale = 0.6
    #             font_thickness = 1
    #
    #             text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    #
    #             text_pt = (x1, y1 - 3)
    #             text_color = [255, 255, 255]
    #
    #             cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
    #             cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
    #                         cv2.LINE_AA)
    # sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss#sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

    return num_dets_to_consider,object_class_index,coordinatesOfObjects,all_Bounding_Box

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
                     lambda i,j: crowd_bbox_iou_cache[i,j].item(),
                     lambda i: box_scores[i], box_indices),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(),
                     lambda i,j: crowd_mask_iou_cache[i,j].item(),
                     lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def YolactEvaluateImage(net:Yolact, frame):
    frame=torch.from_numpy(frame).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    num_dets_to_consider,object_class_index,coordinatesOfObjects,all_Bounding_Box = prep_display(preds, frame, None, None, undo_transform=False)
    return num_dets_to_consider,object_class_index,coordinatesOfObjects,all_Bounding_Box

import time



def CalculateDistance(net:Yolact):
    #MIdas init
    transform,optimize,device,model = MidasnetINIT() #midasnet init done, Midasnet input ready
    #MIdas init

    #setup of ip webcam
    URL = "http://192.168.0.103:8080/video"
    cam = cv2.VideoCapture(URL)
    #setup of ip webcam
    objectsToFindList=RecognizeSpeechAndReturnObjectsAndIndexs()
    # objectsToFindList = [66,77,65]
    fpsLimit = 20  # displays the frame rate every 1 second
    counter = 0

    for DesiredObjectIndex in objectsToFindList:
        # moeez includes start :
        Str ="Desired object I am looking for is "
        Str += str(cfg.dataset.class_names[DesiredObjectIndex-1]) + " ,."
        TextToSpeech(Str) # this will will tell the user about the current disred object in search
        cv2.waitKey(5000)
        # moeez includes end
        prevSideHorizontal = -1  # for direction
        prevSideVertical = -1  # for direction
        while True:
            objectIndexRelative = DesiredObjectIndex - 1  # point where find will be here
            check, frame = cam.read()
            cv2.imshow('IPWebcam', frame)   #my webcam
            num_dets_to_consider,object_class_index,coordinatesOfObjects,all_Bounding_Box = YolactEvaluateImage(net, frame)
            #object_class_index.index(objectIndexRelative)
            try:
                if objectIndexRelative in object_class_index:

                    img_depth = MidasImageDistanceEvaluation(frame, transform, optimize, device, model)
                    avg_SeqWise = AverageObjectDistanceCalculation(num_dets_to_consider, coordinatesOfObjects,
                                                                   img_depth)
                    # to find object in comparision to others
                    indexNumberOfDesiredObject=object_class_index.index(objectIndexRelative)
                    hurdles=CalculateRelativeDistance(objectIndexRelative, object_class_index, avg_SeqWise)
                    prevSideHorizontal, prevSideVertical=CalculateDirection(frame, all_Bounding_Box, prevSideHorizontal, prevSideVertical,indexNumberOfDesiredObject,hurdles)
                    counter += 1
                    print("if counter\n\nframe end",counter)
                    if counter == fpsLimit:
                        counter = 0
                        print("if counter breaks", counter)

                        break
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    counter += 1
                    print("else counter ",counter)
                    if counter == fpsLimit:
                        counter = 0
                        print("else counter breaks", counter)

                        break
                    print("Desired object was not found")
            except:
                print("Exception while true")
                break;
    #moeez's edit : i moved the 3 lines on the bottem out of the for loop
    print("all object to be found is done")
    cam.release()  # After the loop release the cap object
    cv2.destroyAllWindows()  # Destroy all the windows

def CalculateRelativeDistance(objectIndexRelative,object_class_index,avg_SeqWise):
    indexRelativeObject=object_class_index.index(objectIndexRelative)
    i=0
    str="hurdles are "
    for avg in avg_SeqWise:
        if avg_SeqWise[indexRelativeObject]<avg: #if toFindObject is further so find object with larger value comparision to toFindObject
            # if cfg.cfg.dataset.class_names[object_class_index[i]] == "dining table" : continue
            str += cfg.dataset.class_names[object_class_index[i]]
            str += " , "
            print("avg:",avg," in comparision:",avg_SeqWise[indexRelativeObject], "cobj:",cfg.dataset.class_names[object_class_index[i]],"with index",object_class_index[i]," in comparision:", cfg.dataset.class_names[object_class_index[indexRelativeObject]],"with index",object_class_index[indexRelativeObject])
        i+=1
    print("average is compared -CalculateRelativeDistance")
    return str

def CalculateDirection(img,all_Bounding_Box,prevSideHorizontal, prevSideVertical,indexNumberOfDesiredObject,hurdles):

    x,y=all_Bounding_Box[indexNumberOfDesiredObject][0][0],all_Bounding_Box[indexNumberOfDesiredObject][0][1]
    w,h=all_Bounding_Box[indexNumberOfDesiredObject][1][0]-all_Bounding_Box[indexNumberOfDesiredObject][0][0],all_Bounding_Box[indexNumberOfDesiredObject][1][1]-all_Bounding_Box[indexNumberOfDesiredObject][0][1]
    prevSideHorizontal, prevSideVertical= DirectionOfDesiredObject(img,prevSideHorizontal,prevSideVertical,x,y,w,h,hurdles)
    return prevSideHorizontal, prevSideVertical

def DirectionOfDesiredObject(img,prevSideHorizontal,prevSideVertical,x,y,w,h,hurdles):
    hT, wT, cT = img.shape
    wT_of_eachside = (wT / 3)#width screen of 3 parts
    hT_of_eachside = (hT / 3)#height screen of 3 parts

    wTarr = [0] * 3#init
    hTarr = [0] * 3#init
    # width screen proportions
    wTarr[0] = 0
    wTarr[1] = wTarr[0] + wT_of_eachside
    wTarr[2] = wTarr[1] + wT_of_eachside

    # height screen proportions
    hTarr[0] = 0
    hTarr[1] = hTarr[0] + hT_of_eachside
    hTarr[2] = hTarr[1] + hT_of_eachside

    mx = x + int(w / 2)#middle x
    my = y + int(h / 2)#middle y

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.rectangle(img, (x + int(w / 2), y + int(h / 2)), (x + int(w / 2) + 1, y + int(h / 2) + 1), (255, 0, 255), 2)

    #direction x
    cv2.rectangle(img, (int(wTarr[0]), int(hTarr[0])), (int(wTarr[1]), img.shape[1]), (255, 0, 0), 2)
    cv2.rectangle(img, (int(wTarr[1]), int(hTarr[0])), (int(wTarr[2]), img.shape[1]), (0, 0, 255), 2)

    # direction y
    cv2.rectangle(img, (int(wTarr[0]), int(hTarr[0])), (wT, int(hTarr[1])), (0, 0, 255), 2)
    cv2.rectangle(img, (int(wTarr[0]), int(hTarr[1])), (wT, int(hTarr[2])), (0, 0, 255), 2)

    # condition for checking if before entry so prevhorizontal and prevVertical will be same to unique numbers so it shouldn't repeat itself
    # left 0 inner 0 1 2
    # mid 1 inner 0 1 2
    # inner 0 1 2
    if mx >= wTarr[0] and mx <= wTarr[1]:  # left unique prevhorizontal 0
        if my >= hTarr[0] and my <= hTarr[1] and (
                prevSideVertical == -1 or (not (prevSideHorizontal == 0 and prevSideVertical == 0))):  # left up 0
            prevSideHorizontal = 0
            prevSideVertical = 0
            str = "Object is in screen right top"
            #moeez edit start :
            str += ', '
            str += hurdles  # concating the 2 strings
            str += '. '
            TextToSpeech(str)
            #moeez edit close

            #TextToSpeech(str)
            #TextToSpeech(hurdles)
        elif my >= hTarr[1] and my <= hTarr[2] and (
                prevSideVertical == -1 or (not (prevSideHorizontal == 0 and prevSideVertical == 1))):  # left mid
            prevSideHorizontal = 0
            prevSideVertical = 1
            str = " Object is in screen right mid"
            # moeez edit start :
            str += ', '
            str += hurdles  # concating the 2 strings
            str += '. '
            TextToSpeech(str)
            # moeez edit close

            # TextToSpeech(str)
            # TextToSpeech(hurdles)

        else:
            if my >= hTarr[2] and my <= hT and (
                    prevSideVertical == -1 or (not (prevSideHorizontal == 0 and prevSideVertical == 2))):  # left down
                prevSideHorizontal = 0
                prevSideVertical = 2
                str = "Object is in screen right down"

                # moeez edit start :
                str += ', '
                str += hurdles  # concating the 2 strings
                str += '. '
                TextToSpeech(str)
                # moeez edit close

                # TextToSpeech(str)
                # TextToSpeech(hurdles)
    elif mx >= wTarr[1] and mx <= wTarr[2]:  # mid unique prevhorizontal 1
        if my >= hTarr[0] and my <= hTarr[1] and (
                prevSideVertical == -1 or (not (prevSideHorizontal == 1 and prevSideVertical == 0))):  # mid up
            prevSideHorizontal = 1
            prevSideVertical = 0
            str = "Object is in screen mid top"
            # moeez edit start :
            str += ', '
            str += hurdles  # concating the 2 strings
            str += '. '
            TextToSpeech(str)
            # moeez edit close

            # TextToSpeech(str)
            # TextToSpeech(hurdles)
        elif my >= hTarr[1] and my <= hTarr[2] and (
                prevSideVertical == -1 or (not (prevSideHorizontal == 1 and prevSideVertical == 1))):  # mid mid
            prevSideHorizontal = 1
            prevSideVertical = 1
            str = "Object is in screen mid"
            # moeez edit start :
            str += ', '
            str += hurdles  # concating the 2 strings
            str += '. '
            TextToSpeech(str)
            # moeez edit close

            # TextToSpeech(str)
            # TextToSpeech(hurdles)
        else:
            if my >= hTarr[2] and my <= hT and (
                    prevSideVertical == -1 or (not (prevSideHorizontal == 1 and prevSideVertical == 2))):  # mid down
                prevSideHorizontal = 1
                prevSideVertical = 2
                str = "Object is in screen mid down"

                # moeez edit start :
                str += ', '
                str += hurdles  # concating the 2 strings
                str += '. '
                TextToSpeech(str)
                # moeez edit close
                #
                # TextToSpeech(str)
                # TextToSpeech(hurdles)
    else:
        if mx >= wTarr[2] and mx <= wT:  # right
            if my >= hTarr[0] and my <= hTarr[1] and (
                    prevSideVertical == -1 or (not (prevSideHorizontal == 2 and prevSideVertical == 0))):  # right up
                prevSideHorizontal = 2
                prevSideVertical = 0
                str = "Object is in screen left top"

                # moeez edit start :
                str += ', '
                str += hurdles  # concating the 2 strings
                str += '. '
                TextToSpeech(str)
                # moeez edit close

                # TextToSpeech(str)
                # TextToSpeech(hurdles)
            elif my >= hTarr[1] and my <= hTarr[2] and (
                    prevSideVertical == -1 or (not (prevSideHorizontal == 2 and prevSideVertical == 1))):  # right mid
                prevSideHorizontal = 2
                prevSideVertical = 1
                str = "Object is in screen left mid"
                # moeez edit start :
                str += ', '
                str += hurdles  # concating the 2 strings
                str += '. '
                TextToSpeech(str)
                # moeez edit close

                # TextToSpeech(str)
                # TextToSpeech(hurdles)
            else:
                if my >= hTarr[2] and my <= hT and (prevSideVertical == -1 or (
                not (prevSideHorizontal == 2 and prevSideVertical == 2))):  # right down
                    prevSideHorizontal = 2
                    prevSideVertical = 2
                    str = "Object is in screen left down"

                    # moeez edit start :
                    str += ', '
                    str += hurdles  # concating the 2 strings
                    str += '. '
                    TextToSpeech(str)
                    # moeez edit close

                    # TextToSpeech(str)
                    # TextToSpeech(hurdles)
    cv2.imshow("Direction image",img)
    return prevSideHorizontal, prevSideVertical


from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:Yolact, path:str, out_path:str=None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    
    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:

        vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that 
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        cv2.imshow(path, frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                            % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                
                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
                
                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()


    extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None
            
            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args =  [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                
                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)
                
                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence)-1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0
            
            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')
    
    cleanup_and_exit()

def evaluate(net:Yolact, dataset, train_mode=False):
    # MidasNetWrapper()
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    CalculateDistance(net)


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()



if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')

        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)

