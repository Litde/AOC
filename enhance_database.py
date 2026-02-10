import os
from PIL import Image, ImageOps
from tqdm import tqdm

# INPUT_DIR = "D:\Polibuda\Sezon_2_Semestr_2\AOC\input"
INPUT_DIR = "D:\Polibuda\Sezon_2_Semestr_2\AOC\input\\triangle"
# OUTPUT_DIR = "D:\Polibuda\Sezon_2_Semestr_2\AOC\enhanced_input"
OUTPUT_DIR = "D:\Polibuda\Sezon_2_Semestr_2\AOC\enhanced_images\\triangle"
PADDING = 20          # pixels
PADDING_COLOR = (0, 0, 0)  # black padding (RGB)
PADDING_COLOR2 = (255, 255, 255)  # white padding (RGB)
PADDING_COLOR3 = (0, 128, 0) # green padding (RGB)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def process_dataset(input_dir, output_dir):
    for class_name in tqdm(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Create output folder
        out_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            image_name = filename.split(".")[0]
            extension = filename.split(".")[-1]

            new_filename1 = image_name + "_padded1" + "." + extension
            new_filename2 = image_name + "_padded2" + "." + extension
            new_filename3 = image_name + "_padded3" + "." + extension

            input_image_path = os.path.join(class_path, filename)

            with Image.open(input_image_path) as img:
                img = img.convert("RGB")

                # Save original
                original_save_path = os.path.join(out_dir, filename)
                img.save(original_save_path)

                for i, color in enumerate([PADDING_COLOR, PADDING_COLOR2, PADDING_COLOR3]):
                    new_filename = image_name + "_padded" + str(i) + "." + extension

                    # Add padding
                    padded_img = ImageOps.expand(
                        img,
                        border=PADDING,
                        fill=color
                    )

                    # Save padded
                    padded_save_path = os.path.join(out_dir, new_filename)
                    padded_img.save(padded_save_path)

                padding = PADDING * 2
                for i, color in enumerate([PADDING_COLOR, PADDING_COLOR2, PADDING_COLOR3]):
                    new_filename = image_name + "_padded2" + str(i) + "." + extension

                    # Add padding
                    padded_img = ImageOps.expand(
                        img,
                        border=padding,
                        fill=color
                    )

                    # Save padded
                    padded_save_path = os.path.join(out_dir, new_filename)
                    padded_img.save(padded_save_path)

    print("Processing completed.")


if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)