from PIL import Image
import os

def image_crop(image_path, save_path, threshold):
    # Open the image
    image = Image.open(image_path)
    print(image_path)

    # Get the size of the image
    width, height = image.size

    # Loop through each pixel of the image and find the first non-white pixel
    left = -1
    for y in range(height): # start at left top
        for x in range(width):
            if any(pix < threshold for pix in image.getpixel((x, y))[:3]):
                if x < left or left == -1:
                    left = x
                break
        else:
            continue

    # Loop through each pixel of the image from right to left and find the first non-white pixel
    right = - 1
    for y in range(height): # start at right top
        for x in range(width - 1, -1, -1):
            if any(pix < threshold for pix in image.getpixel((x, y))[:3]):
                if x > right or right == -1:
                    right = x
                break
            else:
                continue
    
    # Loop through each pixel of the image from top to bottom and find the first non-white pixel
    top = -1
    for x in range(width): # start at left top
        for y in range(height):
            if any(pix < threshold for pix in image.getpixel((x,y ))[:3]):
                if y < top or top == -1:
                    top = y
                break
            else:
                continue

    # Loop through each pixel of the image from bottom to top and find the first non-white pixel
    bottom = -1
    for x in range(width): # start at left bottom
        for y in range(height - 1, -1, -1):
            if any(pix < threshold for pix in image.getpixel((x, y))[:3]):
                if y > bottom or bottom == -1:
                    bottom = y
                break
            else:
                continue
    
    # Crop the image using the found borders
    if left > 20:
        left -= 20
    else:
        left = 0
    
    if right < width - 20:
        right += 20
    else:
        right = width
    
    if top > 20:
        top -= 20
    else:
        top = 0
    
    if bottom < height -20:
        bottom += 20
    else:
        bottom == height

    # print("left =", left)
    # print("right =", right)
    # print("top =", top)
    # print("bottom =", bottom)
    
    image = image.crop((left, top, right, bottom))

    # Save the cropped image
    image.save(save_path)


directory = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Illustrations/MABA/Analysis/Influence ratio test parameter/'
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        save_path = directory +'Cropped/' +filename
        image_crop(image_path, save_path, 240)