import cv2
import numpy as np
from math import sqrt, exp
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model


# Sigmoid函数用于调整相似度得分
def sigmoid(x, slope=10, threshold=0.5):
    score = 1 / (1 + exp(-slope * (x - threshold)))
    if score > 0.9:
        score = score - 0.085
    elif 0.9 > score > 0.6:
        score = score - 0.075
    elif 0.5 > x > 0.2:
        score = x
    elif score < 0.2:
        score = 0.2

    return score


def draw_boundary_lines(image, pts, colors):
    # 根据输入的 pts 和颜色，分别为每条边画线
    for i in range(len(pts)):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[(i + 1) % len(pts)])  # 取出下一个点形成线段
        color = colors[i]
        cv2.line(image, pt1, pt2, color, 2)


def compute_mean_color(image, mask):
    """计算掩码区域的平均颜色"""
    return cv2.mean(image, mask=mask)[:3]  # 返回 BGR 的均值


def SimilarityforWoodland(image, roi_coords, threshold, is_global=False, magnification=3):
    # ——————————————————————————图片处理部分——————————————————————————
    np_img = image.astype('uint8')

    y_max, x_max, _ = np_img.shape  # 获取图像的高、宽

    # 四边形区域的顶点坐标
    pts = np.array(roi_coords, dtype=np.int32)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    # 创建与图像大小相同的掩码，并在四边形区域内填充白色
    mask = np.zeros(np_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # ——————————————————————————相似度计算——————————————————————————
    # 计算四边形 ROI 区域的均值颜色
    roi_mean = cv2.mean(np_img, mask=mask)[:3]  # BGR 均值

    if is_global:
        # 如果 is_global=True，使用全图作为比较区域
        surrounding_image = np_img  # 全图
        surrounding_mask = np.ones(np_img.shape[:2], dtype=np.uint8) * 255

        # 排除 ROI 区域
        cv2.fillPoly(surrounding_mask, [pts], 0)
    else:
        # 计算包围该四边形的矩形边界
        x, y, w, h = cv2.boundingRect(pts)

        # 框选原始图像长宽 magnification 倍的区域作为比较区域
        x_surrounding = max(x - magnification * w, 0)  # 考虑边界区域
        y_surrounding = max(y - magnification * h, 0)
        x_surrounding_max = min(x_max, x + (magnification + 1) * w)
        y_surrounding_max = min(y_max, y + (magnification + 1) * h)

        # 提取周围区域
        surrounding_image = np_img[y_surrounding:y_surrounding_max, x_surrounding:x_surrounding_max]

        # 在周围区域创建一个与 surrounding_image 相同大小的掩码，并排除 ROI 区域
        surrounding_mask = np.ones(surrounding_image.shape[:2], dtype=np.uint8) * 255
        cv2.fillPoly(surrounding_mask, [pts - [x_surrounding, y_surrounding]], 0)

    # 计算周围区域的均值颜色（排除 ROI 区域或全图中非 ROI 部分）
    surrounding_mean = cv2.mean(surrounding_image, mask=surrounding_mask)[:3]

    # 归一化均值到 [0,1] 范围
    mean_b, mean_g, mean_r = [c / 255.0 for c in roi_mean]
    mean_b2, mean_g2, mean_r2 = [c / 255.0 for c in surrounding_mean]

    # 计算欧氏距离
    delta = (mean_b2 - mean_b) ** 2 + (mean_g2 - mean_g) ** 2 + (mean_r2 - mean_r) ** 2
    similarity = 1 - sqrt(delta)

    # 使用 sigmoid 函数来调节相似度分数
    similarity = sigmoid(similarity)

    # ——————————————————————————计算每条边的相似度——————————————————————————
    line_list = []  # 保存每条边的颜色
    for i in range(len(pts)):
        # 获取当前边的两个端点
        pt1 = pts[i]
        pt2 = pts[(i + 1) % len(pts)]

        # 计算当前边的中点
        mid_point = np.mean([pt1, pt2], axis=0).astype(int)

        # 定义该边附近的区域，使用 magnification 来确定区域大小
        x, y = mid_point
        w = abs(pt2[0] - pt1[0]) if abs(pt2[0] - pt1[0]) != 0 else 100
        h = abs(pt2[1] - pt1[1]) if abs(pt2[1] - pt1[1]) != 0 else 100

        x_start = int(max(x - w, 0))
        y_start = int(max(y - h, 0))
        x_end = int(min(x_max, x + w))
        y_end = int(min(y_max, y + h))

        # 提取该边附近的区域
        surrounding_image = np_img[y_start:y_end, x_start:x_end]

        # 创建掩码，将该边所围区域排除
        surrounding_mask = np.ones(surrounding_image.shape[:2], dtype=np.uint8) * 255
        roi_in_surrounding = pts - [x_start, y_start]  # 将 ROI 转换到相对坐标
        cv2.fillPoly(surrounding_mask, [roi_in_surrounding], 0)

        # 计算该边附近区域的均值颜色
        surrounding_mean = compute_mean_color(surrounding_image, surrounding_mask)

        # 归一化均值到 [0,1] 范围
        mean_b, mean_g, mean_r = [c / 255.0 for c in roi_mean]
        mean_b2, mean_g2, mean_r2 = [c / 255.0 for c in surrounding_mean]

        # 计算欧氏距离
        delta = (mean_b2 - mean_b) ** 2 + (mean_g2 - mean_g) ** 2 + (mean_r2 - mean_r) ** 2
        line_similarity = 1 - sqrt(delta)

        # 使用 sigmoid 函数来调节相似度分数
        line_similarity = sigmoid(line_similarity)

        # 判断是否需要调整（相似度低于阈值）
        if line_similarity < threshold:
            line = [pt1, pt2]
            line_list.append(line)

    return similarity, line_list


def SimilarityforDesert(image, roi_coords, is_global=False, magnification=2):
    def piecewise(x):
        if x >= 0.9:
            x = max(x - 0.055, 0.90)
        elif 0.9 > x >= 0.7:
            x = min(x + 0.085, 0.90)
        elif 0.7 > x >= 0.5:
            x = max(x - 0.06, 0.5)
        elif 0.5 > x > 0.2:
            x = x
        elif x <= 0.2:
            x = 0.2
        return x

    # ————————————————————————————————————HSV————————————————————————————————————
    # ------------------------------处理ROI区域------------------------------
    # 创建一个与原图大小相同的全黑掩膜
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 绘制 ROI 多边形区域在掩膜上
    roi_coords = np.array(roi_coords, dtype=np.int32)
    cv2.fillPoly(mask, [roi_coords], 255)

    # 使用掩膜提取多边形区域的 HSV 值
    masked_hsv = cv2.bitwise_and(image, image, mask=mask)

    # 分离 H、S、V 通道
    h_channel, s_channel, v_channel = cv2.split(masked_hsv)

    # 计算掩膜中非零像素的均值（即多边形区域的 H、S、V 平均值）
    h_mean = np.mean(h_channel[mask == 255])
    s_mean = np.mean(s_channel[mask == 255])
    v_mean = np.mean(v_channel[mask == 255])

    if not is_global:
        # ------------------------------处理周围区域------------------------------
        x, y, w, h = cv2.boundingRect(np.array(roi_coords, dtype=np.int32))
        h_max, w_max = image.shape[:2]
        x_surrounding = max(x - magnification * w, 0)
        y_surrounding = max(y - magnification * h, 0)
        x_surrounding_max = min(w_max, x + (magnification + 1) * w)
        y_surrounding_max = min(h_max, y + (magnification + 1) * h)

        surrounding_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        surrounding_coords = np.array([[x_surrounding, y_surrounding], [x_surrounding, y_surrounding_max],
                                       [x_surrounding_max, y_surrounding_max], [x_surrounding_max, y_surrounding]],
                                      dtype=np.int32)
        cv2.fillPoly(surrounding_mask, [surrounding_coords], 255)

        masked_surrounding_hsv = cv2.bitwise_and(image, image, mask=surrounding_mask)
        masked_surrounding_hsv -= masked_hsv  # 排除ROI区域

        h_surrounding, s_surrounding, v_surrounding = cv2.split(masked_surrounding_hsv)

        # 计算掩膜中非零像素的均值
        h_surrounding_mean = np.mean(h_surrounding[surrounding_mask == 255])
        s_surrounding_mean = np.mean(s_surrounding[surrounding_mask == 255])
        v_surrounding_mean = np.mean(v_surrounding[surrounding_mask == 255])

        H_distance = abs(h_surrounding_mean - h_mean)
        S_distance = abs(s_surrounding_mean - s_mean)
        V_distance = abs(v_surrounding_mean - v_mean)
    else:
        # ------------------------------处理整个图像区域------------------------------
        h_channel, s_channel, v_channel = cv2.split(image)

        # 不进行局部周围区域的处理，直接用整个图像的HSV均值作为参考
        h_mean_global = np.mean(h_channel)
        s_mean_global = np.mean(s_channel)
        v_mean_global = np.mean(v_channel)

        # 计算整体的HSV均值与ROI区域的HSV均值的差异
        H_distance = abs(h_mean_global - h_mean)
        S_distance = abs(s_mean_global - s_mean)
        V_distance = abs(v_mean_global - v_mean)

    Similarity_HSV = 1 - (H_distance / 180 + S_distance / 255 + V_distance / 255)

    # ————————————————————————————————————网络的方法————————————————————————————————————
    # 加载预训练的 ResNet50 模型，并去掉最后的分类层
    base_model = ResNet50(weights="imagenet", pooling="avg", include_top=False)

    def extract_features(image):
        # 图像预处理
        img_resized = cv2.resize(image, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # 使用 ResNet50 的预处理方法

        # 提取特征
        layer_name = 'conv3_block3_out'  # 例如，第二个阶段的第一个块输出
        intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
        features = intermediate_layer_model.predict(img_array)
        return features.flatten()

    # 提取 ROI 的边界坐标
    x, y, w, h = cv2.boundingRect(np.array(roi_coords, dtype=np.int32))

    x_min = min(coord[0] for coord in roi_coords)
    x_max = max(coord[0] for coord in roi_coords)
    y_min = min(coord[1] for coord in roi_coords)
    y_max = max(coord[1] for coord in roi_coords)

    # 裁剪 ROI 区域
    img_roi = image[y_min:y_max, x_min:x_max]

    # 计算周围区域
    h_max, w_max = image.shape[:2]
    x_surrounding = max(x - magnification * w, 0)
    y_surrounding = max(y - magnification * h, 0)
    x_surrounding_max = min(w_max, x + (magnification + 1) * w)
    y_surrounding_max = min(h_max, y + (magnification + 1) * h)
    surrounding_area = image[y_surrounding:y_surrounding_max, x_surrounding:x_surrounding_max]

    # 提取特征
    roi_features = extract_features(img_roi)
    if is_global:
        surrounding_features = extract_features(image)
    else:
        surrounding_features = extract_features(surrounding_area)

    # 计算余弦相似度
    similarity = np.dot(roi_features, surrounding_features) / (
            np.linalg.norm(roi_features) * np.linalg.norm(surrounding_features)
    )

    # 归一化并应用分段函数
    Similarity_Net = piecewise((similarity + 1) / 2)

    final_Similarity = piecewise(min(Similarity_HSV, Similarity_Net))

    # ————————————————————————————————————计算每条边的相似度————————————————————————————————————
    def get_parallel_region_within_roi(pt1, pt2, pt3, pt4, thickness):
        # 计算 pt1->pt2、 pt2->pt3、pt1->pt4、pt3->pt4
        pt1_pt2 = np.array(pt2) - np.array(pt1)
        pt2_pt3 = np.array(pt3) - np.array(pt2)
        pt3_pt4 = np.array(pt4) - np.array(pt3)
        pt1_pt4 = np.array(pt4) - np.array(pt1)

        # 归一化这些向量
        pt1_pt2 = pt1_pt2 / np.linalg.norm(pt1_pt2)
        pt2_pt3 = pt2_pt3 / np.linalg.norm(pt2_pt3)
        pt3_pt4 = pt3_pt4 / np.linalg.norm(pt3_pt4)
        pt1_pt4 = pt1_pt4 / np.linalg.norm(pt1_pt4)

        # 计算偏移量
        pt1_pt2_point1 = pt1 + pt1_pt4 * thickness
        pt1_pt2_point2 = pt2 + pt2_pt3 * thickness
        region1 = np.array([pt1, pt2, pt1_pt2_point2, pt1_pt2_point1], dtype=np.int32)

        pt2_pt3_point1 = pt2 - pt1_pt2 * thickness
        pt2_pt3_point2 = pt3 + pt3_pt4 * thickness
        region2 = np.array([pt2, pt3, pt2_pt3_point2, pt2_pt3_point1], dtype=np.int32)

        pt3_pt4_point1 = pt3 - pt2_pt3 * thickness
        pt3_pt4_point2 = pt4 - pt1_pt4 * thickness
        region3 = np.array([pt3, pt4, pt3_pt4_point2, pt3_pt4_point1], dtype=np.int32)

        pt1_pt4_point1 = pt1 + pt1_pt2 * thickness
        pt1_pt4_point2 = pt4 - pt3_pt4 * thickness
        region4 = np.array([pt4, pt1, pt1_pt4_point1, pt1_pt4_point2], dtype=np.int32)

        return [region1, region2, region3, region4]

    pts = np.array(roi_coords, dtype=np.int32)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    thickness = 15
    # 获取平行区域的四个顶点
    region = get_parallel_region_within_roi(pts[0], pts[1], pts[2], pts[3], thickness)

    line_list = []
    for i in region:
        coords = i

        # 创建一个与原图大小相同的全黑掩膜
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask, [coords], 255)

        # 使用掩膜提取多边形区域的 HSV 值
        masked_hsv = cv2.bitwise_and(image, image, mask=mask)

        # 分离 H、S、V 通道
        h_channel, s_channel, v_channel = cv2.split(masked_hsv)

        # 计算掩膜中非零像素的均值（即多边形区域的 H、S、V 平均值）
        h_mean = np.mean(h_channel[mask == 255])
        s_mean = np.mean(s_channel[mask == 255])
        v_mean = np.mean(v_channel[mask == 255])

        # ------------------------------处理周围区域------------------------------
        x, y, w, h = cv2.boundingRect(np.array(roi_coords, dtype=np.int32))
        h_max, w_max = image.shape[:2]
        x_surrounding = max(x - magnification * w, 0)
        y_surrounding = max(y - magnification * h, 0)
        x_surrounding_max = min(w_max, x + (magnification + 1) * w)
        y_surrounding_max = min(h_max, y + (magnification + 1) * h)

        surrounding_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        surrounding_coords = np.array([[x_surrounding, y_surrounding], [x_surrounding, y_surrounding_max],
                                       [x_surrounding_max, y_surrounding_max], [x_surrounding_max, y_surrounding]],
                                      dtype=np.int32)
        cv2.fillPoly(surrounding_mask, [surrounding_coords], 255)

        masked_surrounding_hsv = cv2.bitwise_and(image, image, mask=surrounding_mask)
        masked_surrounding_hsv -= masked_hsv  # 排除ROI区域

        h_surrounding, s_surrounding, v_surrounding = cv2.split(masked_surrounding_hsv)

        # 计算掩膜中非零像素的均值
        h_surrounding_mean = np.mean(h_surrounding[surrounding_mask == 255])
        s_surrounding_mean = np.mean(s_surrounding[surrounding_mask == 255])
        v_surrounding_mean = np.mean(v_surrounding[surrounding_mask == 255])

        H_distance = abs(h_surrounding_mean - h_mean)
        S_distance = abs(s_surrounding_mean - s_mean)
        V_distance = abs(v_surrounding_mean - v_mean)

        Similarity_HSV = 1 - (H_distance / 180 + S_distance / 255 + V_distance / 255)

        # 判断是否需要调整（相似度低于阈值）
        if Similarity_HSV < 0.7:
            line = [coords[0], coords[1]]
            line_list.append(line)

    return final_Similarity, line_list


def SimilarityforRadar(image, roi_coords, is_global=True, magnification=4):
    def piecewise(x):
        if x >= 0.9:
            x = max(x - 0.055, 0.90)
        elif 0.9 > x >= 0.7:
            x = min(x + 0.085, 0.90)
        elif 0.7 > x > 0.2:
            x = max(x - 0.08, 0.2)
        elif x <= 0.2:
            x = 0.2
        return x

    np_img = image.astype('uint8')

    y_max, x_max, _ = np_img.shape  # 获取图像的高、宽

    # 四边形区域的顶点坐标
    pts = np.array(roi_coords, dtype=np.int32)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    # 创建与图像大小相同的掩码，并在四边形区域内填充白色
    mask = np.zeros(np_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 使用掩膜提取多边形区域的 HSV 值
    masked_hsv = cv2.bitwise_and(image, image, mask=mask)

    # 分离 H、S、V 通道
    _, _, v_channel = cv2.split(masked_hsv)

    v_median = 0.8 * np.median(v_channel[mask == 255])
    v_mean = np.mean(v_channel[mask == 255])

    if not is_global:
        x, y, w, h = cv2.boundingRect(np.array(roi_coords, dtype=np.int32))
        h_max, w_max = image.shape[:2]
        x_surrounding = max(x - magnification * w, 0)
        y_surrounding = max(y - magnification * h, 0)
        x_surrounding_max = min(w_max, x + (magnification + 1) * w)
        y_surrounding_max = min(h_max, y + (magnification + 1) * h)

        surrounding_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        surrounding_coords = np.array([[x_surrounding, y_surrounding], [x_surrounding, y_surrounding_max],
                                       [x_surrounding_max, y_surrounding_max], [x_surrounding_max, y_surrounding]],
                                      dtype=np.int32)
        cv2.fillPoly(surrounding_mask, [surrounding_coords], 255)

        masked_surrounding_hsv = cv2.bitwise_and(image, image, mask=surrounding_mask)
        masked_surrounding_hsv -= masked_hsv  # 排除ROI区域

        h_surrounding, s_surrounding, v_surrounding = cv2.split(masked_surrounding_hsv)

        v_mean_surrounding = np.mean(v_surrounding[surrounding_mask == 255])  # 雷达图仅有V通道

        dist_radar = min(
            1 - (abs(v_mean_surrounding - v_median) / v_mean_surrounding),
            1 - (abs(v_mean_surrounding - v_mean) / v_mean_surrounding)
        )
    else:
        _, _, v_channel_image = cv2.split(image)

        v_mean_images = np.mean(v_channel_image)

        dist_radar = min(
            1 - (abs(v_mean_images - v_median) / v_mean_images),
            1 - (abs(v_mean_images - v_mean) / v_mean_images)
        )

    return piecewise(dist_radar)


def SimilarityforInfrared(image, roi_coords, is_global=False, magnification=2):
    def piecewise(x):
        if x >= 0.9:
            x = max(x - 0.055, 0.90)
        elif 0.9 > x >= 0.7:
            x = max(x - 0.05, 0.7)
        elif 0.7 > x >= 0.5:
            x = max(x - 0.08, 0.5)
        elif 0.5 > x > 0.2:
            x = x
        elif x <= 0.2:
            x = 0.2
        return x

    np_img = image.astype('uint8')

    y_max, x_max, _ = np_img.shape  # 获取图像的高、宽

    # 四边形区域的顶点坐标
    pts = np.array(roi_coords, dtype=np.int32)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    x, y, w, h = cv2.boundingRect(pts)

    # 创建与图像大小相同的掩码，并在四边形区域内填充白色
    mask = np.zeros(np_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 计算四边形 ROI 区域的均值颜色
    roi_mean = cv2.mean(np_img, mask=mask)[:3]

    if not is_global:
        # 框选原始图像长宽 magnification 倍的区域作为比较区域
        x_surrounding = max(x - magnification * w, 0)  # 考虑边界区域
        y_surrounding = max(y - magnification * h, 0)
        x_surrounding_max = min(x_max, x + (magnification + 1) * w)
        y_surrounding_max = min(y_max, y + (magnification + 1) * h)

        # 提取周围区域
        surrounding_image = np_img[y_surrounding:y_surrounding_max, x_surrounding:x_surrounding_max]

        # 在周围区域创建一个与 surrounding_image 相同大小的掩码，并排除 ROI 区域
        surrounding_mask = np.ones(surrounding_image.shape[:2], dtype=np.uint8) * 255
        cv2.fillPoly(surrounding_mask, [pts - [x_surrounding, y_surrounding]], 0)

        # 计算周围区域的均值颜色（排除 ROI 区域或全图中非 ROI 部分）
        surrounding_mean = cv2.mean(surrounding_image, mask=surrounding_mask)[:3]

        # 计算 RGB 均值的距离
        dist_infrared = 1 - min(
            (np.max(roi_mean) + 1e-5) / (np.min(surrounding_mean) + 1e-4), 4
        ) * 0.25
    else:
        img_mean = np.mean(image, axis=(0, 1))

        # 计算 RGB 均值的距离
        dist_infrared = 1 - min(
            (np.max(roi_mean) + 1e-5) / (np.min(img_mean) + 1e-4), 4
        ) * 0.25

    dist_infrared = round(dist_infrared, 4)

    return piecewise(dist_infrared)


if __name__ == '__main__':
    # ————————————————————————————————————————————可见光————————————————————————————————————————————
    # image = cv2.imread('/home/saki/Desktop/相似度/20231107-ysy-luoche/KJ_0100_202311081435.tif')
    # roi_coords = [(2288, 1753), (2307, 1736), (2360, 1800), (2343, 1816)]  # 裸车：0.57
    # image = cv2.imread('/home/saki/Desktop/相似度/20231109-ysy-jiawang02/KJ_0100_202311091559.tif')
    # roi_coords = [(2267, 1757), (2297, 1734), (2370, 1812), (2332, 1835)]  # 架网：0.88
    # roi_coords = [[2270, 1748], [2318, 1705], [2390, 1807], [2330, 1847]]

    # image = cv2.imread('../KJ_0100_202311142358.tif')
    # roi_coords = [(2215, 2258), (2353, 2267), (2349, 2399), (2206, 2396)]   # 林地网：0.57
    # roi_coords = [(3301, 1494), (3438, 1483), (3441, 1617), (3301, 1629)]   # 荒漠网：0.87
    # roi_coords = [(2380, 1596), (2453, 1701), (2360, 1778), (2281, 1676)]   # 雪地网：0.2

    # similarity, line_list = SimilarityforDesert(image, roi_coords, is_global=False)

    # print(similarity)
    # print(line_list)

    # w, h = image.shape[:2]

    # ————————————————————————————————————————————雷达————————————————————————————————————————————
    # image = cv2.imread('/home/saki/Desktop/相似度/20231107-ysy-luoche/LD_0150_202311071331.jpg')   # 裸车：0.2
    # image = cv2.imread('/home/saki/Desktop/相似度/20231109-ysy-jiawang02/LD_0150_202311091053.jpg')    # 架网：0.29
    # roi_coords = [(2288, 1753), (2307, 1736), (2360, 1800), (2343, 1816)]

    image = cv2.imread('/home/saki/Desktop/相似度/20231112-sxw-jiawang-wu-qing/LD_0150_202311121036.jpg')
    roi_coords = [(2215, 2258), (2353, 2267), (2349, 2399), (2206, 2396)]
    # roi_coords = [(3301, 1494), (3438, 1483), (3441, 1617), (3301, 1629)]
    # roi_coords = [(2380, 1596), (2453, 1701), (2360, 1778), (2281, 1676)]
    # roi_coords = [[1350, 2700], [1500, 2700], [1500, 2850], [1350, 2850]]
    # roi_coords = [[2281, 1672], [2386, 1593], [2455, 1703], [2356, 1778]]
    # roi_coords = [[2157, 990], [2265, 987], [2261, 1054], [2165, 1060]]

    # image = cv2.resize(image, (w, h))
    similarity = SimilarityforRadar(image, roi_coords)
    print(similarity)

    # ————————————————————————————————————————————红外————————————————————————————————————————————
    # image = cv2.imread('/home/saki/Desktop/相似度/20231107-ysy-luoche/HW_0150_202311082325.tif')   # 裸车：0.52
    # image = cv2.imread('/home/saki/Desktop/相似度/20231109-ysy-jiawang02/HW_0150_202311092128.tif')    # 架网：0.7

    # image = cv2.resize(image, (w, h))
    # similarity = SimilarityforInfrared(image, roi_coords, is_global=False)
    # print(similarity)
