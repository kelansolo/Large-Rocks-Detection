import os
import shutil
import argparse
import random

# Create command line parser
parser = argparse.ArgumentParser(
                    prog='SplitMaker',
                    description='Randomly split the given dataset into train, validation and test datasets using provided fractions of the dataset (full set by default). The sum of the fractions can be less or equal than 1 but not greater',
                    epilog='---')


parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('-f', '--fractions', default= [0.6,0.2,0.2], type=float,  nargs=3, help = 'Order of fractions : train validation test. Defult is 0.6 0.2 0.2. Sum of the 3 can not be larger than 1')
parser.add_argument('-s', '--size', default= 0, type=int, help = 'How many images to take (e.x. if you want to test the pipeline on 10 images). \n Default is *all*')



def files_empty(output_dir):
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is empty
            if os.path.isfile(file_path) and os.path.getsize(file_path) != 0:
                return False
    return True


def format_dirs(input_dir,output_dir): #Format directories
    input_dir = os.path.normpath(input_dir)
    output_dir = os.path.normpath(output_dir)

    # Create the directories
    os.makedirs(output_dir + "/train/images", exist_ok=False)
    os.makedirs(output_dir + "/val/images", exist_ok=False)
    os.makedirs(output_dir + "/test/images", exist_ok=False)

    return input_dir,output_dir

def split(input_dir, output_dir, fract, dataset_size):
    files = os.listdir(input_dir)
    # Shuffle the list of image filenames
    random.seed(42)
    random.shuffle(files)

    train_size = int(len(files) * fract[0])
    val_size = int(len(files) * fract[1])
    test_size = int(len(files) * fract[2])

    if dataset_size != 0:
        files = files[0:dataset_size]

    for i, f in enumerate(files):
        if i < train_size:
            dest_folder ="train"
        elif i < train_size + val_size:
            dest_folder = "val"
        elif i < train_size + val_size + test_size:      
            dest_folder = "test"
        else:
            continue
        shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, os.path.join(dest_folder+"/images", f)))
        

    # get actual file sizes
    test_size = len(os.listdir(os.path.join(output_dir, "test/images")))
    train_size = len(os.listdir(os.path.join(output_dir, "train/images")))
    val_size = len(os.listdir(os.path.join(output_dir, "val/images")))

    return train_size, val_size, test_size

def main():
    args = parser.parse_args()
    fract = args.fractions
    dataset_size = args.size
    input_dir = args.input_dir
    output_dir = args.output_dir

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