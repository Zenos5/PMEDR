import matplotlib.pyplot as plt 
import io 
import argparse
from PIL import Image 

def plot_losses(path, title, x_label, y_label):
    values = []
    with open(path, 'r') as fp:
        values = fp.readlines()
        values = [float(v) for v in values]
    plt.title(title) 
    plt.xlabel(x_label) 
    plt.ylabel(y_label) 
    plt.plot(values) 
    fig = plt.gcf() 
    return fig
    
def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return img 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="checkpoints/", help = "path to checkpoint")
    args = parser.parse_args()
    

    # plot NLL figure
    # print("path:", args.path + "train_loss.txt")

    # fig = plot_losses(args.path + "train_loss.txt", "Training Loss per Epoch", "Epoch", "Loss")
    # img = fig2img(fig)
    # img.save('results/_ibem/Losses_Graph.png') 

    # fig = plot_losses(args.path + "val_prec.txt", "Precision per Epoch", "Epoch", "Precision")
    # img = fig2img(fig)
    # img.save('results/_ibem/Precision_Graph.png') 

    # fig = plot_losses(args.path + "val_rec.txt", "Recall per Epoch", "Epoch", "Recall")
    # img = fig2img(fig)
    # img.save('results/_ibem/Recall_Graph.png')

    # fig = plot_losses(args.path + "val_mAP.txt", "mAP per Epoch", "Epoch", "mAP")
    # img = fig2img(fig)
    # img.save('results/_ibem/mAP_Graph.png') 

    # Checkpoint 100
    fig = plot_losses(args.path + "train_loss_100.txt", "Training Loss per Epoch", "Epoch", "Loss")
    img = fig2img(fig)
    img.save('results/_ibem/Losses_Graph_100.png') 

    # fig = plot_losses(args.path + "val_prec_100.txt", "Precision per Epoch", "Epoch", "Precision")
    # img = fig2img(fig)
    # img.save('results/_ibem/Precision_Graph_100.png') 

    # fig = plot_losses(args.path + "val_rec_100.txt", "Recall per Epoch", "Epoch", "Recall")
    # img = fig2img(fig)
    # img.save('results/_ibem/Recall_Graph_100.png')

    # fig = plot_losses(args.path + "val_mAP_100.txt", "mAP per Epoch", "Epoch", "mAP")
    # img = fig2img(fig)
    # img.save('results/_ibem/mAP_Graph_100.png') 