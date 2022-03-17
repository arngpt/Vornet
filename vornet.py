import argparse
import os
import shutil
import cv2
import glob
import tensorflow as tf
import numpy as np
from time import time
from scipy import stats
import matplotlib.pyplot as plt

class VOR():

    def __init__(self, project_name):

        self.project_name = project_name
        self.project_path = os.path.join('Data', self.project_name)
        self.background_path = os.path.join(self.project_path, 'background')
        self.input_video_path = os.path.join(self.project_path, 'input_video')
        self.extracted_frames_path = os.path.join(self.project_path, 'extracted_frames')
        self.object_removed_frames = os.path.join(self.project_path, 'object_removed_frames')
        self.output_frames_path = os.path.join(self.project_path, 'output_frames')
        self.output_video_path = os.path.join(self.project_path, 'output_video')
        self.set_confidence = 0.5
        self.set_nms_threshold = 0.1

    def create_tree(self):

        os.makedirs(self.project_path)
        os.makedirs(self.input_video_path)
        os.makedirs(self.background_path)
        os.makedirs(self.extracted_frames_path)
        os.makedirs(self.object_removed_frames)
        os.makedirs(self.output_frames_path)
        os.makedirs(self.output_video_path)

        if os.path.exists(self.project_path) and os.path.exists(self.input_video_path) and os.path.exists(self.extracted_frames_path) and os.path.exists(self.object_removed_frames) and os.path.exists(self.background_path) and os.path.exists(self.output_frames_path) and os.path.exists(self.output_video_path) :
            print ("All directories created successfully")
            self.video_path = args.video_path
            copy_check = shutil.copy(self.video_path, self.input_video_path)
            if copy_check:
                print ("Placed input video in the directory")
            else:
                print("Error in copying video")

        else:
            ("Error occured in creating directories")

    def extract_frames(self):

        print ("Extracting frames...")
        videos = list()
        videos.append(os.listdir(self.input_video_path))
        full_video_name = videos[0][0]
        video = cv2.VideoCapture(os.path.join(self.input_video_path, full_video_name))
        count = 0

        while (video.isOpened()):

            ret, frame = video.read()

            if ret == True:
                cv2.imwrite(os.path.join(self.extracted_frames_path, "input_{}.jpg".format(count)), frame)  
                count += 1
                if (count % 100 == 0):
                    print('Read %d frame: ' % count, ret)
            else:
                break
            
        video.release()
        cv2.destroyAllWindows()

    def get_frames(self, ip_dir):
        image_list = list()
        image_from_folder = list()

        for file in glob.glob(ip_dir + '/*.jpg'):
            image_list.append(os.path.split(file)[1])

        image_list.sort(key = lambda x: int(x[6:-4]))

        for i in range(len(image_list)):
            frame = os.path.join(ip_dir, image_list[i])
            img = cv2.imread(frame) 
            image_from_folder.append(img)
            
        print ('Total frames: ',len(image_from_folder))
        return image_from_folder

    def get_model_config(self, model):
        
        if model == 'yolo':
            weights = './Models/yolo/yolov3.weights'
            configr = './Models/yolo/yolov3.cfg'
            classes = './Models/yolo/yolov3.txt'

        elif model == 'mask-rcnn':
            weights = './Models/mask-rcnn/mask_rcnn_weights.pb'
            configr = './Models/mask-rcnn/mask_rcnn_config.pbtxt'
            classes = './Models/mask-rcnn/mask_rcnn_classes.txt'

        else:
            print ('Unknown model')

        return weights, configr, classes

    def background_predict(self, method):

        if not os.path.exists(os.path.join(self.background_path, 'background_' + str(method) + '.jpg')):
            all_frames = self.get_frames(self.extracted_frames_path)
            if method == 'median':
                est_bg = np.median(all_frames, axis=0)
                background = os.path.join(self.background_path, 'background_' + str(method) + '.jpg')
                pred = cv2.imwrite(background, est_bg)
                if pred:
                    print ('Predicted background saved!')
                else:
                    print ('Error in predicting and saving background')

            elif method == 'mode':
                est_bg = stats.mode(all_frames, axis=0)
                background = os.path.join(self.background_path, 'background_' + str(method) + '.jpg')
                pred = cv2.imwrite(background, est_bg[0][0]) 
                if pred:
                    print ('Predicted background saved!')
                else:
                    print ('Error in predicting and saving background')

            else:
                print ('Unknown background prediction method')

        else:
            print ('Backgroung already predicted')
            background = os.path.join(self.background_path, 'background_' + str(method) + '.jpg')
        return background

    def remove_object(self, image, classes, class_id, confidence, x, y, x_plus_w, y_plus_h, target, i):
        
        image[y:y_plus_h, x:x_plus_w] = [0,0,0]
        filename = os.path.join(self.object_removed_frames, 'frame_' + str(target) + '_' + str(i) + '.jpg')
        cv2.imwrite(filename, image)
        print('Object found in frame {}. Save frame for processing'.format(i))
        return True, filename

    def inpaint_yolo(self, image, x, y, x_plus_w, y_plus_h, backg, ipf):

        full_img = cv2.imread(image)
        full_bkg = cv2.imread(backg)
        inp = full_bkg[y:y_plus_h, x:x_plus_w]
        full_img[y:y_plus_h, x:x_plus_w] = inp
        filename = os.path.join(self.output_frames_path, 'out_' + str(ipf) + '.jpg')
        res = cv2.imwrite(filename, full_img)
        print ('Frame {} process complete!'.format(ipf))
        return res
    
    def inpaint_maskrcnn(self, image, mask, x, y, x_plus_w, y_plus_h, backg, ipf):
        
        full_img = cv2.imread(image)
        full_bkg = cv2.imread(backg)
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 255:
                    full_img[i, j] = full_bkg[i, j]
        filename = os.path.join(self.output_frames_path, 'out_' + str(ipf) + '.jpg')
        res = cv2.imwrite(filename, full_img)
        print ('Frame {} process complete!'.format(ipf))
        return res

    def model_train(self, model):
        weights = model[0]
        configr = model[1]
        class_f = model[2]

        classes = None
        with open(class_f, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet(weights, configr)
        return classes, net
    
    def yolo(self, model, bg):
        
        print ('Model chosen: yolo')
        classes, net = self.model_train(model)
        if not args.target in classes:
            print('Such a target object cannot be detected. Please select object from given classes only: ')
            print(classes)
            exit()

        all_frames = self.get_frames(self.extracted_frames_path)
        for ipf in range(len(all_frames)):

            image = all_frames[ipf]
            Width = image.shape[1]
            Height = image.shape[0]
            scale = 0.00392

            # detect objects
            blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

            # feed input image
            net.setInput(blob)

            # get ALL output layers ie detect all objects in input image
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.set_confidence, self.set_nms_threshold)

            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0] - 10
                y = box[1] - 10
                w = box[2] + 20
                h = box[3] + 20

                label = str(classes[class_ids[i]])
                if label == args.target:
                    obj, filename = self.remove_object(image, classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), args.target, ipf)       
                    if obj:
                        inp = self.inpaint_yolo(filename, round(x), round(y), round(x+w), round(y+h), bg, ipf)
                        if inp:
                            break

                else:
                    print('Sorry! Object not found in frame {}. Object detected as {}'.format(ipf, str(label)))
                    filename = os.path.join(self.output_frames_path, 'out_' + str(ipf) + '.jpg')
                    cv2.imwrite(filename, image)
                
    def mask_rcnn(self, model, bg):
        
        print ('Model chosen: mask-rcnn')
        classes, net = self.model_train(model)
        if not args.target in classes:
            print('Such a target object cannot be detected. Please select object from given classes only: ')
            print(classes)
            exit()
    
        all_frames = self.get_frames(self.extracted_frames_path)
        for ipf in range(len(all_frames)):

            image = all_frames[ipf]
            Width = image.shape[1]
            Height = image.shape[0]
            
            # detect objects
            blob = cv2.dnn.blobFromImage(image, True, crop=False)

            # feed input image
            net.setInput(blob)

            # get ALL output layers ie detect all objects in input image
            (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

            for i in range(0, boxes.shape[2]):
                classID = int(boxes[0, 0, i, 1])
                confidence = boxes[0, 0, i, 2]
                
                if confidence > self.set_confidence:
                    
                    clone = image.copy()
                    box = boxes[0, 0, i, 3:7] * np.array([Width, Height, Width, Height])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxW = endX - startX
                    boxH = endY - startY

                    mask = masks[i, classID]
                    mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > self.set_nms_threshold)
                    
                    roi = clone[startY:endY, startX:endX]
                    visMask = (mask * 255).astype("uint8")
                    
                    binMask = np.zeros((image.shape[0], image.shape[1]))
                    binMask[startY:endY, startX:endX] = visMask

                    roi = roi[mask]

                    label = str(classes[classID])

                    if label == args.target:
                        blended = ((0 * roi)).astype("uint8")

                        clone[startY:endY, startX:endX][mask] = blended

                        filename = os.path.join(self.object_removed_frames, 'frame_' + str(args.target) + '_' + str(ipf) + '.jpg')
                        cv2.imwrite(filename, clone)
                        
                        print('Object found in frame {}. Save frame for processing. Object detected is {}'.format(ipf, label))
                        
                        inp = self.inpaint_maskrcnn(filename, binMask, startX, startY, endX, endY, bg, ipf)
                        if inp:
                            break

                    else:
                        filename = os.path.join(self.output_frames_path, 'out_' + str(ipf) + '.jpg')
                        cv2.imwrite(filename, image)
                    
    def out_video(self):

        videos = list()
        videos.append(os.listdir(self.input_video_path))
        full_video_name = videos[0][0]
        video = cv2.VideoCapture(os.path.join(self.input_video_path, full_video_name))
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = video.get(cv2.CAP_PROP_FPS)

        video.release()      
        print ('Preparing video...')
        frames = self.output_frames_path
        video = os.path.join(self.output_video_path, 'output.mp4')

        frame_list = list()
        filenames = list()

        for file in glob.glob(frames + '/*.jpg'):
            filenames.append(os.path.split(file)[1])

        filenames.sort(key = lambda x: int(x[4:-4]))

        for i in range(len(filenames)):
            frame = os.path.join(frames, filenames[i])
            img = cv2.imread(frame)
            frame_list.append(img)
            h, w = img.shape[:2]
        size = (w, h)

        out = cv2.VideoWriter(video,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(frame_list)):
            out.write(frame_list[i])
        out.release()


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', action='store', help='Give the project name to be created/worked on')
    parser.add_argument('action', choices=['create', 'extract', 'train', 'output_video'], action='store', help='create - Create new working directory.\
    extract - Extract frames from input video.\
    train - Load the model to be used and background prediction technique.\
    output_video - Output Processing')
    parser.add_argument('-v', '--video_path', action='store', help='Give path of input video')
    parser.add_argument('-m', '--model', action='store', help='Give model configuration. yolo or mask-rcnn', default='yolo')
    parser.add_argument('-b', '--background', action='store', help='Give background generation technique. median or mode', default='mode')
    parser.add_argument('-t', '--target', action='store', help='Enter target object')
    args = parser.parse_args()

    if args.action == 'create':

        vor_var = VOR(args.project_name)
        project_path = os.path.join('./Data', args.project_name)

        if not os.path.exists(project_path):
            vor_var.create_tree()
            
        else:
            print ("Project already exists")

    elif args.action == 'extract':

        vor_var = VOR(args.project_name)
        if (len(os.listdir(vor_var.extracted_frames_path))) == 0:
            vor_var.extract_frames()
        else:
            print ("Frames may already be extracted before")

    elif args.action == 'train':

        vor_var = VOR(args.project_name)

        if (len(os.listdir(vor_var.background_path))) < 2:
            bg = vor_var.background_predict(args.background)

        else:
            print ("Background already predicted")
            if args.background == 'median':
                bg = os.path.join(vor_var.background_path, 'background_median.jpg')
            elif args.background == 'mode':
                bg = os.path.join(vor_var.background_path, 'background_mode.jpg')
            print('Chosen background: ', args.background)

        if args.target == None:
            print ('Please specify target!')
            exit()

        model_def = vor_var.get_model_config(args.model)
        if args.model == 'yolo':
            vor_var.yolo(model_def, bg)

        elif args.model == 'mask-rcnn':
            vor_var.mask_rcnn(model_def, bg)

    elif args.action == 'output_video':

        vor_var = VOR(args.project_name)
        vor_var.out_video()