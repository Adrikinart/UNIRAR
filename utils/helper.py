import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_color(map_,img):
    if map_.ndim == 3:
        map_ = map_.squeeze(0)
    map_ = map_.detach().cpu().numpy()

    map_ = cv2.normalize(map_, None, 0, 255, cv2.NORM_MINMAX)
    map_ = cv2.resize(map_, (img.shape[1], img.shape[0]))
    map_ = map_.astype(np.uint8)

    map_color = cv2.applyColorMap(map_, cv2.COLORMAP_JET)
    map_color = cv2.cvtColor(map_color, cv2.COLOR_BGR2RGB)
    map_img = cv2.addWeighted(img, 0.6, map_color, 0.4, 0)

    return map_, map_img


def show_saliency(img, maps= {}, details= None):


    titles= ['Image']
    saliency_colors = [img]
    saliency_colors_images = [img]

    for key, v in maps.items():
        saliency_color, saliency_color_img = apply_color(v, img)
        titles.append(key)
        saliency_colors.append(saliency_color)
        saliency_colors_images.append(saliency_color_img)

    titles_details= []
    colors_details= []
    colors_details_images= []

    if details is not None:
        details= details.squeeze(0)
        for i in range(details.shape[0]):
            detail_color, detail_color_img = apply_color(details[i, :, :], img)
            titles_details.append(f"Detail {i}")
            colors_details.append(detail_color)
            colors_details_images.append(detail_color_img)

    plt.figure(figsize=(24,18))

    rows =  2
    cols = max(len(saliency_colors) , len(colors_details))

    if len(titles_details) != 0:
        rows += 2

    for i in range(cols):
        plt.subplot(rows,cols,i + 1)
        plt.imshow(saliency_colors[i])
        plt.axis('off')
        plt.title(titles[i])


        plt.subplot(rows,cols,cols + i + 1)
        plt.imshow(saliency_colors_images[i])
        plt.axis('off')
        plt.title(titles[i])

    # show details
    for i in range(len(titles_details)):
        plt.subplot(rows,cols,2*cols + i + 1)
        plt.imshow(colors_details[i])
        plt.axis('off')
        plt.title(titles_details[i])

        plt.subplot(rows,cols,3*cols + i + 1)
        plt.imshow(colors_details_images[i])
        plt.axis('off')
        plt.title(titles_details[i])
    

    # output_dir = "./outputs/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_path = os.path.join(output_dir, f"image_{index}_{args.name}_finetune_{args.finetune.lower()}_threshold_{args.threshold}.jpeg")
    # plt.savefig(output_path)

    # plt.show()
