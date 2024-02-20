from PIL import Image

def cut_area(image_path, start_coordinates, resolution):
    # Open the image
    img = Image.open(image_path)

    # Calculate the pixels for the 4x4m area based on the given resolution
    area_size = (4 * resolution, 4 * resolution)

    # Calculate the end coordinates based on the start coordinates and area size
    end_coordinates = (start_coordinates[0] + area_size[0], start_coordinates[1] + area_size[1])

    # Crop the image
    cropped_img = img.crop((*start_coordinates, *end_coordinates))

    # Return the cropped image
    return cropped_img

def cut_area_center(image_path, resolution):
    """
    The resolution parameter in the function is critical for determining the real-world dimensions of the cropped area. 
    The resolution, usually measured in pixels per inch (PPI) or dots per inch (DPI), signifies the number of pixels 
    that fit into a 1-inch line. In the context of this function, resolution represents the number of pixels per meter 
    in the image. Therefore, if you want a 4x4 meter area from the image, you multiply 4 by the resolution to get the 
    dimensions of the area in pixels.
    """
    
    # Open the image
    img = Image.open(image_path)

    # Get the size of the image
    width, height = img.size

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Calculate the size of the 4x4m area in pixels
    area_size = (4 * resolution, 4 * resolution)

    # Calculate the start coordinates for the crop so that the cropped area will be centered
    start_coordinates = (center[0] - area_size[0] // 2, center[1] - area_size[1] // 2)

    # Calculate the end coordinates based on the start coordinates and area size
    end_coordinates = (start_coordinates[0] + area_size[0], start_coordinates[1] + area_size[1])

    # Crop the image
    cropped_img = img.crop((*start_coordinates, *end_coordinates))

    # Return the cropped image
    return cropped_img