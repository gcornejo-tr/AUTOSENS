import random
import numpy as np
from modelB import DualHeadResNet18
import torch
from utils import *
from dataloader import *
import cv2
import matplotlib.pyplot as plt



def create_1d_gaussian(mean, std_dev, x_range=(-1, 50), points=1000):
    """Generates a 1D Gaussian distribution along a continuous axis."""
    x = np.linspace(x_range[0]-mean, x_range[1]+mean, points)
    gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return x, gaussian


if __name__ == "__main__":


    name_seq = '/LIGHTING/LOW/10p'
    name_seq = '/FOG/LOW/05p'

    lateral_seq = "lateralset" + name_seq
    frontal_seq = "dataset" + name_seq

    FOLDER_PATH = frontal_seq

    PATH = "trained/modelJZ8/modelB_68.pth"    

    K_passes = 25
    
    sequence_list = get_sequences(FOLDER_PATH)
    tv_sequence_list =[ i for i in sequence_list if 'LIGHTING' in i]+[ i for i in sequence_list if 'FOG/LOW' in i]
    train_list = random.sample(tv_sequence_list, int(0.8*len(tv_sequence_list)))
    val_list  = list(set(tv_sequence_list) - set(train_list))
    test_list   = list(set(sequence_list) - set(tv_sequence_list))

    test_list = frontal_seq

    print(len(train_list), len(val_list), len(test_list))

    test_dataset = MyDataset(FOLDER_PATH, test_list)
    test_loader = DataLoader(test_dataset , batch_size = 8, shuffle=False )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadResNet18(num_classes=1).to(device)
    checkpoint = torch.load(PATH, map_location=device)
    state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    video_writer = None
    

    for nb, dict in enumerate(test_loader):
        bs =dict["image_tensor"].shape[0]

        img_batch = dict["image_tensor"].to(device)
        data_annot = dict["annotation_tensor"].to(device)

        t_cls = torch.where(data_annot[:,4]==0, 1.0, 0.0)
        t_dis = (torch.where(data_annot[:,4]==0, data_annot[:,5], torch.zeros_like(data_annot[:,5])))
        
        multiple_Dpred = []
        multiple_Cpred = []

    
        for i in range(K_passes): 

            with torch.no_grad():
                pred_class, pred_distance = model(img_batch)

            pred_class = pred_class.squeeze(-1).squeeze(-1)  
            multiple_Cpred.append(pred_class.squeeze(-1).squeeze(-1))
            pred_distance = pred_distance.squeeze(-1).squeeze(-1) 
            pred_distance = pred_distance
            multiple_Dpred.append(pred_distance)

            print(pred_class)

        multiple_Dpred = torch.stack(multiple_Dpred, axis=1).to('cpu').detach().cpu().numpy()
        multiple_Cpred = torch.stack(multiple_Cpred, axis=1).to('cpu').detach().cpu().numpy()

        for i in range(multiple_Dpred.shape[0]):

            fig, ax = plt.subplots()
            pred_class_np = multiple_Cpred[i]
            predC_mean = np.mean(pred_class_np)

            if predC_mean>0.5:
                pred_distances_np = multiple_Dpred[i]
                pred_mean = np.mean(pred_distances_np)
                pred_std = np.std(pred_distances_np)
            else:
                pred_mean = 0.0
                pred_std = 50.0
    
            gt_mean = t_dis[i].detach().cpu().numpy()

            print(gt_mean, pred_mean )

            x_axis, pred_gauss = create_1d_gaussian(mean=pred_mean, std_dev=pred_std)
            _, gt_gauss = create_1d_gaussian(mean=gt_mean, std_dev=0.01)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                x_axis,
                pred_gauss,
                label=f"Predicted Distance Gaussian (Pred {pred_mean:.2f} [m])",
                color="blue",
                linewidth=2,
            )

            ax.axvline(
                x=gt_mean, color="green", linestyle=":", label=f"GT Mean ({gt_mean:.2f} [m])"
            )
            
            ax.set_xlim(-1, 50 )
            ax.set_ylim(-0.0, 0.75)

            ax.set_title("1D Distance Gaussian: Prediction vs Ground Truth")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Probability Density")
            ax.legend()
            ax.grid(True, alpha=0.3)           
            ax.yaxis.tick_right()         
            ax.yaxis.set_label_position("right")
            ax.invert_xaxis() 
            fig.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            height, width, _ = frame_bgr.shape

            output_video_path = f"analysis/video_example_B.mp4"

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_video_path, fourcc, 20 , (width, height)
                )

            video_writer.write(frame_bgr)
            plt.close(fig)

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {output_video_path}")


