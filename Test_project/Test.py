import argparse
import torch


from utils import (
    get_test_loader,
    get_transform,
    check_accuracy,
    load_checkpoint,
    save_predictions_as_imgs
)


from Model_test.VET_FF_Net.VET_FF_Net_Model import VEET_FF_Net


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='LUNG', help='experiment_name, please use all capital letters')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--test_image_save', type=bool, default='False', help='whether to save the test picture')
parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as png')
parser.add_argument('--cuda', type=str, default="cuda:0", help='cuda')
parser.add_argument('--pin_memory', type=bool,  default=False, help='DataLoader.pin_memory')
# parser.add_argument('--drop_list', type=bool,  default=True, help='DataLoader.drop_last')

args = parser.parse_args()

if __name__ =='__main__':

    DEVICE = args.cuda

    model =VEET_FF_Net()
    model.to(device=DEVICE)

    # model.load_state_dict(torch.load('Model_test/best_model_new.pth', map_location=DEVICE))        #use .pth
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)     #use checkpoint

    #Dataset
    TEST_IMG_DIR = "data/{}/val_images".format(args.dataset)
    TEST_MASK_DIR = "data/{}/val_masks".format(args.dataset)

    #DataLoader
    test_loader=get_test_loader(TEST_IMG_DIR,
                                TEST_MASK_DIR,
                                args.batch_size,
                                get_transform(),
                                args.num_classes,
                                args.pin_memory,
                                )

    check_accuracy(test_loader, model, device=DEVICE)

    if args.test_image_save:
        save_predictions_as_imgs(
            test_loader, model, folder=args.test_save_dir, device=DEVICE)





