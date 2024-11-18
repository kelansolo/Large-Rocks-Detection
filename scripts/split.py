import os
import shutil
import argparse
import random

# Create command line parser
parser = argparse.ArgumentParser(
                    prog='SplitMaker',
                    description='Makes split with train, validation and test images using provided a given fraction of the dataset (full set by default)',
                    epilog='---')


parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('-f', '--fractions', default= [0.6,0.2,0.2], type=float,  nargs=3)
parser.add_argument('-s', '--size', default= 0, type=int)

args = parser.parse_args()

fract = args.fractions
dataset_size = args.size
input_dir = args.input_dir
output_dir = args.output_dir

def files_empty(output_dir):
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is empty
            if os.path.isfile(file_path) and os.path.getsize(file_path) != 0:
                return False
    return True


def format_dirs(input_dir,output_dir): #Format directories
    if input_dir[-1] != "/":
        input_dir = input_dir + "/"
    if output_dir[-1] != "/":
        output_dir = output_dir + "/"

    # Create the directories
    os.makedirs(output_dir + "/train/", exist_ok=False)
    os.makedirs(output_dir + "/val/", exist_ok=False)
    os.makedirs(output_dir + "/test/", exist_ok=False)

    return input_dir,output_dir

def split(input_dir, output_dir, fract, dataset_size):
    files = os.listdir(input_dir)
    # Shuffle the list of image filenames
    random.seed(42)
    random.shuffle(files)
    if dataset_size != 0:
        files = files[0:dataset_size]

    # determine the number of images for each set
    train_size = int(len(files) * fract[0])
    val_size = int(len(files) * fract[1])
    test_size = int(len(files) * fract[2])
    # print("--- Split debug ---")
    # print("Train size: {}".format(train_size))
    # print("Validation size: {}".format(val_size))
    # print("Test size: {}".format(test_size))    

    for i, f in enumerate(files):
        if i < train_size:
            dest_folder ="train"
            # shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, os.path.join(dest_folder, f)))
        elif i < train_size + val_size:
            dest_folder = "val"
            # shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, os.path.join(dest_folder, f)))
        elif i < train_size + val_size + test_size:      
            dest_folder = "test"
        else:
            continue
        shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, os.path.join(dest_folder, f)))
        

    # get actual file sizes
    test_size = len(os.listdir(os.path.join(output_dir, "test")))
    train_size = len(os.listdir(os.path.join(output_dir, "train")))
    val_size = len(os.listdir(os.path.join(output_dir, "val")))

    return train_size, val_size, test_size

def main():
    global input_dir, output_dir, fract, dataset_size
    if fract[0]+fract[1]+fract[2] > 1:
        print("Sum of fractiions can't be larger than 1")
    else:
        try:
            input_dir,output_dir = format_dirs(input_dir,output_dir)
        except FileExistsError:
            response = input("Output folders already exist, would you like to overwite them (y/n): ").strip().lower()
            if response == "y":
                shutil.rmtree(output_dir)
                input_dir,output_dir = format_dirs(input_dir,output_dir)
            else:
                return

        except OSError as e:
            print(f"An error occurred: {e}")
            return
        
        train_size, val_size, test_size = split(input_dir, output_dir, fract, dataset_size)
        print("--- Split finished ---")
        print("Train size: {}".format(train_size))
        print("Validation size: {}".format(val_size))
        print("Test size: {}".format(test_size))    

if __name__ == "__main__":
    main()