import copy

import numpy as np
from keras.models import load_model
import os
import cv2
from zichazijiu import *

from Similarity import SimilarityforWoodland, SimilarityforDesert, SimilarityforRadar, SimilarityforInfrared

magnification = 2
b_global_comparison = False  # True:全图对比，False,区域对比，magnification启用


def listdir(path, ext, list_name, key=None):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, ext, list_name)
        else:
            if file_path.endswith(ext) and key is None:
                list_name.append(file_path)
            elif file_path.endswith(ext) and key is not None and key in file_path:
                list_name.append(file_path)


class Cerect_proposal(object):
    def __init__(
            self,
            model_save_path,
            material_image_path,
            material_angle_path,
            land_key_str,
            geo_type: int,

    ):
        self.embedding_network = load_model(model_save_path)
        self.material_imagevec, self.material_jpffile_list = self._load_material_imagevec(material_image_path,
                                                                                          land_key_str)
        self.material_image_path = material_image_path
        self.material_angle_path = material_angle_path

        self.geo_type = geo_type

        self.descripe = {"1": {}, "2": {}, "3": {}}     # 分别对应可见光、红外、雷达

        self.descripe["1"]["visible"] = "调整轮廓，模仿附近自然背景轮廓，请根据图中标示的边缘线条调整相应的网布边缘形状或在边缘架设仿真树或布设杂草"
        self.descripe["1"]["infrared"] = "布设区域的红外特征与全局红外特征存在明显的明暗差异，需要调整斑块附近支撑变形件，增大网布与目标距离，使网布与目标表面隔开"
        self.descripe["1"]["radar"] = "布设区域的雷达成像特征与全局雷达成像特征存在明显的明暗差异，需要调整"

        self.descripe["2"]["visible"] = "请根据图中标示的边缘线条调整轮廓，或在边缘使用防护贴片，或利用附近地物进行伪装"
        self.descripe["2"]["infrared"] = "布设区域的红外特征与全局红外特征存在明显的明暗差异，调整网面或利用附近地物进行伪装，网面调整时装备部分位置贴合网面，轮廓部位不贴合"
        self.descripe["2"]["radar"] = "布设区域的雷达成像特征与全局雷达成像特征存在明显的明暗差异，利用附近地物进行伪装"

        self.descripe["3"]["visible"] = "请根据图中标示的边缘线条调整轮廓，或在边缘使用防护贴片，或利用附近地物进行伪装"
        self.descripe["3"]["infrared"] = "布设区域的红外特征与全局红外特征存在明显的明暗差异，调整网面或利用附近地物进行伪装"
        self.descripe["3"]["radar"] = "布设区域的雷达成像特征与全局雷达成像特征存在明显的明暗差异，利用附近地物进行伪装"

    def conver_class_to_globalclass(self, classsification_list):
        global_classsification_list = []
        head_jpgfile_list = []
        for item in classsification_list:
            filename = self.material_jpffile_list[item]
            classsification_id = int(filename.split('_')[0].replace(self.material_image_path, "")) - 10000
            global_classsification_list.append(classsification_id)
            head_file_name = filename.replace(filename.split('_')[0] + "_", "")
            head_jpgfile_list.append(head_file_name)
        return global_classsification_list, head_jpgfile_list

    def evaluate_erect_proposal(self, image, img_infrared, img_radar, target_array, evaluate_recv_json):
        h, w, c = image.shape  # 获取输入图像的高度、宽度和通道数
        adjustment_scheme_list = []
        distval_list = []

        for target_json in target_array:
            # 初始化每个目标的方案字典
            scheme_json = {}

            # 取得目标的坐标矩形
            rect = target_json["image_coordinates_point"]
            scheme_json["lines_list"] = [[]]  # 初始化线条列表

            scheme_json["protective_facilities"] = 0
            scheme_json["angle_offset"] = 0
            scheme_json["x_offset"] = 0
            scheme_json["y_offset"] = 0

            pt1, pt2, pt3, pt4 = rect
            x_list = [pt1[0], pt2[0], pt3[0], pt4[0]]
            y_list = [pt1[1], pt2[1], pt3[1], pt4[1]]

            min_x = min(x_list)
            max_x = max(x_list)
            min_y = min(y_list)
            max_y = max(y_list)

            if self.geo_type == 1:
                threshold = 0.7  # 判断相似的阈值，用于判断是否需要更改绘制框线

                # 可见光、林地图像的处理
                similarity, line_list = SimilarityforWoodland(
                    image, rect, threshold,
                    b_global_comparison, magnification
                )  # 返回相似度与要更改的线的集合

                dist_img_LD = round(similarity, 4)  # 与周围的相似度

                # 读取更改的线的坐标，传到可视化界面中
                for line in line_list:
                    pt1, pt2 = line
                    # 将 NumPy 数组转换为 Python 列表
                    pt1 = pt1.tolist()  # 或者使用 tuple(pt1) 转换为元组
                    pt2 = pt2.tolist()
                    line_json = {"x1": pt1[0], "y1": pt1[1], "x2": pt2[0], "y2": pt2[1]}
                    scheme_json["lines_list"][0].append(line_json)

                if len(line_list) > 0:
                    scheme_json["visible_description"] = self.descripe[str(self.geo_type)]["visible"]
                else:
                    scheme_json["visible_description"] = "布设区域的可见光特征与全局可见光特征不存在明显差异，不需要调整"

                scheme_json["visible_discovery_probability"] = dist_img_LD

                distval_list.append(dist_img_LD)
            elif self.geo_type == 2:
                # 可见光、荒漠图像的处理
                similarity, line_list = SimilarityforDesert(
                    image, rect, b_global_comparison, magnification
                )  # 返回相似度与要更改的线的集合
                dist_img_HM = round(similarity, 4)  # 与周围的相似度

                # 读取更改的线的坐标，传到可视化界面中
                for line in line_list:
                    pt1, pt2 = line
                    # 将 NumPy 数组转换为 Python 列表
                    pt1 = pt1.tolist()  # 或者使用 tuple(pt1) 转换为元组
                    pt2 = pt2.tolist()
                    line_json = {"x1": pt1[0], "y1": pt1[1], "x2": pt2[0], "y2": pt2[1]}
                    scheme_json["lines_list"][0].append(line_json)

                if len(line_list) > 0:
                    scheme_json["visible_description"] = self.descripe[str(self.geo_type)]["visible"]
                else:
                    scheme_json["visible_description"] = "布设区域的可见光特征与全局可见光特征不存在明显差异，不需要调整"

                scheme_json["visible_discovery_probability"] = dist_img_HM

                distval_list.append(dist_img_HM)
            elif self.geo_type == 3:
                line_list = zichazijiu(rect, image)
                scheme_json["lines_list"] = [[]]

                for line in line_list:
                    pt1, pt2 = line
                    line_json = {"x1": pt1[0], "y1": pt1[1], "x2": pt2[0], "y2": pt2[1]}
                    scheme_json["lines_list"][0].append(line_json)

                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                width = max_x - min_x
                height = max_y - min_y

                minx = int(max(0, min_x - width * magnification))
                maxx = int(min(w, max_x + width * magnification))
                miny = int(max(0, min_y - height * magnification))
                maxy = int(min(h, max_y + height * magnification))

                rect_img = img_gray[min_y:max_y, min_x:max_x]
                if not b_global_comparison:
                    img_gray = img_gray[miny:maxy, minx:maxx]
                img_val = np.median(img_gray)
                rect_val = np.median(rect_img)

                dist_img_XD = 1 - min((max(rect_val, img_val) + 0.00001) / (min(rect_val, img_val) + 0.0001), 4) * 0.25
                dist_img_XD = round(dist_img_XD, 4)

                scheme_json["visible_discovery_probability"] = dist_img_XD
                if len(line_list) > 0:
                    scheme_json["visible_description"] = self.descripe[str(self.geo_type)]["visible"]
                else:
                    scheme_json["visible_description"] = "布设区域的可见光特征与全局可见光特征不存在明显差异，不需要调整"

                distval_list.append(dist_img_XD)

            # ————————————————————————————————————————————————红外图像的处理————————————————————————————————————————————————
            scheme_json["infrared_discovery_probability"] = -1
            if img_infrared is not None:
                image_infrared = cv2.resize(img_infrared, (w, h))
                similarity_infrafed = SimilarityforInfrared(image_infrared, rect, b_global_comparison, magnification)

                if similarity_infrafed < 0.5:
                    scheme_json["infrared_description"] = self.descripe[str(self.geo_type)]["infrared"]
                else:
                    scheme_json["infrared_description"] = "布设区域的红外特征与全局红外特征不存在明显差异，不需要调整"

                scheme_json["infrared_discovery_probability"] = similarity_infrafed

            # 雷达图像的处理
            scheme_json["radar_discovery_probability"] = -1
            if img_radar is not None:
                image_radar = cv2.resize(img_radar, (w, h))
                similarity_Radar = SimilarityforRadar(image_radar, rect, b_global_comparison, magnification)

                if similarity_Radar < 0.5:  # 当相似度小于0.5，也就是选点区域、全局区域的色值中位数差异2倍以上时，认为需要调整。
                    scheme_json["radar_description"] = self.descripe[str(self.geo_type)]["radar"]
                else:
                    scheme_json["radar_description"] = "布设区域的雷达成像特征与全局雷达成像特征不存在明显差异，不需要调整"

                scheme_json["radar_discovery_probability"] = similarity_Radar

            adjustment_scheme_list.append(scheme_json)

        return target_array, adjustment_scheme_list, distval_list

    def compute_erect_proposal(self, image, rect_list, rect_width, outer_width, outer_height,
                               bool_stationarytarget=False):
        #_compute_material_type
        classsification_list, classsification_distval_list, new_rect_list = self._compute_material_type(image,
                                                                                                        rect_list,
                                                                                                        rect_width,
                                                                                                        outer_width,
                                                                                                        outer_height)
        print("classsification_list", classsification_list)
        print("classsification_distval_list", classsification_distval_list)
        print("new_rect_list", new_rect_list)
        if len(classsification_list) == 0:
            return [], [], image, [], []
        if bool_stationarytarget == False:
            classsification_distval_list, classsification_list, new_rect_list = zip(
                *sorted(zip(classsification_distval_list, classsification_list, new_rect_list), reverse=True))
            classsification_distval_list = list(classsification_distval_list)
            classsification_list = list(classsification_list)
            new_rect_list = list(new_rect_list)

        if len(classsification_distval_list) > 10 and bool_stationarytarget == False:
            classsification_distval_list = classsification_distval_list[0:10]
            classsification_list = classsification_list[0:10]
            new_rect_list = new_rect_list[0:10]
        print("classsification_list", classsification_list)
        print("classsification_distval_list", classsification_distval_list)
        print("new_rect_list", new_rect_list)

        #covert classsification_list to global classsification_list
        global_classsification_list, head_jpgfile_list = self.conver_class_to_globalclass(classsification_list)
        print("global_classsification_list", global_classsification_list)
        print("head_jpgfile_list", head_jpgfile_list)

        #_compute_material_angle
        angel_list, image = self._compute_material_angle(image, global_classsification_list, new_rect_list, outer_width,
                                                         outer_height)
        print("angel_list", angel_list)

        return head_jpgfile_list, angel_list, image, new_rect_list, classsification_distval_list

    def compute_erect_proposal_for_stationarytarget(self, image, stationarytarget_list, rect_width, outer_width,
                                                    outer_height):
        rect_list = []
        for target_item in stationarytarget_list:
            rect = target_item["image_coordinates_point"]
            rect_list.append(rect)
        head_jpgfile_list, angle_list, image, rect_list, classsification_distval_list = self.compute_erect_proposal(
            image, rect_list, rect_width, outer_width, outer_height, True)

        return head_jpgfile_list, angle_list, image, rect_list, classsification_distval_list

    def _compute_material_type(self, image, rect_list, width, output_width, output_height):
        image_list = []
        new_rect_list = []
        h, w, c = image.shape
        for rect_item in rect_list:
            if max(np.array(rect_item).reshape(-1)) == 0:
                continue
            pt1, pt2, pt3, pt4 = rect_item
            lefttop_x, lefttop_y = pt1
            rightbottom_x, rightbottom_y = pt3
            centerrect = [lefttop_x, lefttop_y, rightbottom_x, rightbottom_y]
            new_rect_list.append(centerrect)
            img = self.__reduced_merge_image(image, centerrect, output_width, output_height)
            img = cv2.resize(img, (64, 64))
            image_list.append(img)
        if len(image_list) == 0:
            return [], [], []
        image_list = np.array(image_list)
        vec_image = self.__compute_image_to_vector(image_list)
        dist_list = self._compute_distance_imagevec(vec_image, self.material_imagevec)
        classsification_list, classsification_distval_list = self._compute_distance_similar(dist_list)
        return classsification_list, classsification_distval_list, new_rect_list

    def _compute_material_angle(self, image, classsification_list, new_rect_list, output_width, output_height):
        angle_list = []
        image_list = []
        image_list.append(image)
        for (i, rect) in enumerate(new_rect_list):
            image_array = self.__material_into_image(image_list[0], classsification_list[i], rect)  #同一点位，同一材料，各角度图像

            # __split_two_image
            imgarray_outer, imgarray_center = self.__split_two_image(image_array, rect, output_width, output_height)

            #__compute_image_to_vector
            vec_outer = self.__compute_image_to_vector(imgarray_outer)
            vec_center = self.__compute_image_to_vector(imgarray_center)
            #_compute_sim_imagevec
            dist_list = self._compute_pair_distance_imagevec(vec_outer, vec_center)
            #get max similar
            max_pos, max_val = self._compute_pair_distance_similar(dist_list)
            print("attention-3:", max_pos, max_val)
            angle_list.append(max_pos)
            print("attention-4:", angle_list)
            image_list[0] = image_array[max_pos]
        print("attention-5:", angle_list)
        return angle_list, image_list[0]

    def _compute_distance_imagevec(self, src_vec, cmp_vec):
        x = src_vec
        y = cmp_vec
        dist_list = []
        for i in range(len(x)):
            dist_cmp_to_srcvec = []
            for j in range(len(y)):
                if len(x[i]) != len(y[j]):
                    dist_cmp_to_srcvec.append([-1, -1, -1])
                    continue
                num = float(np.dot(x[i], y[j]))  # 若为行向量则 A * B.T
                denom = np.linalg.norm(x[i]) * np.linalg.norm(y[j])
                cos = num / denom  # 余弦值
                sim = 0.5 + 0.5 * cos  # 归一化

                dist = np.linalg.norm(x[i] - y[j])
                dist = 1.0 / (1.0 + dist)  # 归一化
                dist_avg = sim * 0.7 + dist * 0.3
                dist_cmp_to_srcvec.append([sim, dist, dist_avg])
            dist_list.append(dist_cmp_to_srcvec)
        return dist_list

    def _compute_pair_distance_imagevec(self, src_vec, cmp_vec):
        x = src_vec
        y = cmp_vec
        dist_list = []
        for i in range(len(x)):
            num = float(np.dot(x[i], y[i]))  # 若为行向量则 A * B.T
            denom = np.linalg.norm(x[i]) * np.linalg.norm(y[i])
            cos = num / denom  # 余弦值
            sim = 0.5 + 0.5 * cos  # 归一化

            dist = np.linalg.norm(x[i] - y[i])
            dist = 1.0 / (1.0 + dist)  # 归一化
            dist_avg = sim * 0.7 + dist * 0.3
            dist_list.append([sim, dist, dist_avg])

        return dist_list

    def _compute_distance_similar(self, dist_list):

        classsification_list = []
        classsification_distval_list = []
        for (i, srcvec) in enumerate(dist_list):
            val_list = (np.array(srcvec)[:, 2]).tolist()
            max_val = max(val_list)
            if max_val == -1:
                break
            max_pos = val_list.index(max_val)
            classsification_list.append(max_pos)
            if max_val < 0.35:
                max_val = -2
            classsification_distval_list.append(max_val)

        return classsification_list, classsification_distval_list

    def _compute_pair_distance_similar(self, dist_list):
        classsification_list = []
        classsification_distval_list = []
        val_list = np.array(dist_list)[:, 2].tolist()
        print("attention-1:", val_list)
        max_val = max(val_list)
        #max_pos = val_list.index(max_val)
        for i in range(len(val_list)):
            if val_list[i] == max_val:
                max_pos = int(i)
        print("attention-2:", max_val)
        classsification_list.append(max_pos)
        classsification_distval_list.append(max_val)
        return max_pos, max_val

    def _load_material_imagevec(self, image_path, land_key_str):
        imglist = []
        listdir(image_path, "", imglist)
        imglist.sort(key=lambda x: x.lower())
        imagearray = []
        material_jpgfile_list = []
        for imgpath in imglist:
            if not land_key_str in imgpath:
                continue
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            img = img[:, :, 0:3]
            img = cv2.resize(img, (64, 64))
            imagearray.append(img)
            material_jpgfile_list.append(imgpath)
        imagearray = np.array(imagearray)

        vec = self.__compute_image_to_vector(imagearray)
        return vec, material_jpgfile_list

    def _load_material_imagevec(self, image_path, land_key_str):
        imglist = []
        listdir(image_path, "", imglist)
        imglist.sort(key=lambda x: x.lower())
        imagearray = []
        material_jpgfile_list = []
        for imgpath in imglist:
            if not land_key_str in imgpath:
                continue
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            img = img[:, :, 0:3]
            img = cv2.resize(img, (64, 64))
            imagearray.append(img)
            material_jpgfile_list.append(imgpath)
        imagearray = np.array(imagearray)

        vec = self.__compute_image_to_vector(imagearray)
        return vec, material_jpgfile_list

    def load_material_imagevec_filelist(self, filelist):
        image_path = self.material_image_path
        imglist = []
        listdir(image_path, "", imglist)  #只有jpg
        imglist.sort(key=lambda x: x.lower())
        imagearray = []
        material_jpgfile_list = []
        for filename in filelist:
            if filename == '':
                continue
            land_key_str = filename
            for imgpath in imglist:
                if not land_key_str in imgpath:
                    continue
                img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
                img = img[:, :, 0:3]
                img = cv2.resize(img, (64, 64))
                imagearray.append(img)
                material_jpgfile_list.append(imgpath)
        imagearray = np.array(imagearray)

        vec = self.__compute_image_to_vector(imagearray)
        self.material_imagevec = vec
        self.material_jpffile_list = material_jpgfile_list

    def __reduced_merge_image(self, img, center_rect, output_width, output_height):
        h, w, c = img.shape
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = center_rect
        mid_x = int(0.5 * (left_top_x + right_bottom_x))
        mid_y = int(0.5 * (left_top_y + right_bottom_y))
        output_rect = [max(0, int(mid_x - output_width // 2)), max(0, int(mid_y - output_height // 2)),
                       min(w - 1, int(mid_x + output_width // 2)), min(h - 1, int(mid_y + output_height // 2))]
        img_zero = np.zeros(img.shape, dtype=np.uint8)
        img_out = img[output_rect[1]:output_rect[3], output_rect[0]:output_rect[2], :]

        img_zero[output_rect[1]:output_rect[3], output_rect[0]:output_rect[2], :] = img_out  #把大图装进去

        img_out = cv2.resize(img_out, (right_bottom_x - left_top_x, right_bottom_y - left_top_y))

        img_zero[left_top_y:right_bottom_y, left_top_x:right_bottom_x, :] = img_out  #把小图覆盖大图的center_rect区域

        img_out = img_zero[output_rect[1]:output_rect[3], output_rect[0]:output_rect[2], :]

        return img_out

    def __split_two_image(self, imgarray, center_rect, output_width, output_height):
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = center_rect

        h, w, c = imgarray[0].shape

        mid_x = int(0.5 * (left_top_x + right_bottom_x))
        mid_y = int(0.5 * (left_top_y + right_bottom_y))
        output_rect = [max(0, mid_x - output_width // 2), max(0, mid_y - output_height // 2),
                       min(w - 1, mid_x + output_width // 2), min(h - 1, mid_y + output_height // 2)]

        imgarray_outer = []
        imgarray_center = []
        for i in range(len(imgarray)):
            img_out = imgarray[i][output_rect[1]:output_rect[3], output_rect[0]:output_rect[2], :]
            img_center = imgarray[i][left_top_y:right_bottom_y, left_top_x:right_bottom_x, :]

            img_out = cv2.resize(img_out, (64, 64))
            img_center = cv2.resize(img_center, (64, 64))

            imgarray_center.append(img_center)
            imgarray_outer.append(img_out)

        return np.array(imgarray_outer), np.array(imgarray_center)

    def __compute_image_to_vector(self, image_array):
        if len(image_array) == 0:
            return np.array([])
        vec = self.embedding_network.predict(image_array)
        return vec

    def __material_into_image(self, image, classid, rect):
        imagearray = []
        imagefile_list = []
        listdir(self.material_angle_path, 'npy', imagefile_list, key=str(10000 + classid) + "_")
        for (i, imagefile) in enumerate(imagefile_list):
            #print(imagefile)
            img = np.load(imagefile)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            _, contours, heridency = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox = cv2.boundingRect(contours[0])
            [x, y, bw, bh] = bbox
            img = img[y:y + bh, x:x + bw, :]
            #cv2.imwrite(imagefile.replace(".npy",".jpg"),img)
            binary = binary[y:y + bh, x:x + bw]
            width_new = rect[2] - rect[0]
            height_new = rect[3] - rect[1]
            img = cv2.resize(img, (width_new, height_new))
            binary = cv2.resize(binary, (width_new, height_new))
            image_copy_material = copy.deepcopy(image)
            image_copy = copy.deepcopy(image)
            mask_material = np.zeros(image_copy.shape[:2], dtype=np.uint8)
            mask_material[rect[1]:rect[3], rect[0]:rect[2]] = binary
            mask_bg = cv2.bitwise_not(mask_material)
            # 掩模显示背景
            image_copy_material[rect[1]:rect[3], rect[0]:rect[2], :] = img
            img_material = cv2.bitwise_and(image_copy_material, image_copy_material, mask=mask_material)

            # 掩模显示前景
            img_bg = cv2.bitwise_and(image_copy, image_copy, mask=mask_bg)

            dst = cv2.add(img_material, img_bg)

            imagearray.append(dst)

        imagearray = np.array(imagearray)
        return imagearray


if __name__ == '__main__':
    obj = Cerect_proposal("7.h5", "data/jpg\\", "data/npy\\", "ZB")

    img = cv2.imread("KJ_0300_202203291005.tif", cv2.IMREAD_UNCHANGED)
    img = img[:, :, 0:3]

    # data = {'stationary_target': [{'target_type': 300003, 'target_identifier': 1,
    #                                'image_coordinates_point': [[369, 0], [468, 0], [468, 94], [369, 94]]},
    #                               {'target_type': 300038, 'target_identifier': 2,
    #                                'image_coordinates_point': [[326, 168], [425, 168], [425, 267], [326, 267]]},
    #                               {'target_type': 300039, 'target_identifier': 3,
    #                                'image_coordinates_point': [[207, 282], [306, 282], [306, 381], [207, 381]]}],
    #         'moving_target': [{'type': 300004, 'number': 1}],
    #         'threat_type': {'visible_threat_type': 240092, 'infrared_threat_type': 240113, 'radar_threat_type': 240116},
    #         'geographical_type': 1100, 'visible_image_absolutepath': 'KJ_0300_202203291005.tif',
    #         'infrared_image_absolutepath': '', 'radar_image_absolutepath': '',
    #         'visible_tfw_absolutepath': 'KJ_0300_202203291005.tfw', 'infrared_tfw_absolutepath': '',
    #         'radar_tfw_absolutepath': '', 'light_type': 1, 'only_stationary': 0, 'pretreatment': 0,
    #         'move_target_maxlength': 15, 'camouflage_comb': 'ZB_A_Z_1_30m.jpg', 'target_gap': 50}
    ratio = 0.5 * (0.15 + 0.15)

    rect_width = int(15 // ratio)
    outer_width = int(50 // ratio)
    outer_height = int(50 // ratio)
    head_jpgfile_list, angel_list, image, new_rect_list, classsification_distval_list = obj.compute_erect_proposal(
        img,
        [[[
            4013,
            3729],
            [
                4113,
                3729],
            [
                4113,
                3829],
            [
                4013,
                3829]]],
        rect_width,
        outer_width,
        outer_height,
        bool_stationarytarget=False)
    cv2.imwrite("./out/image_out.jpg", image)
