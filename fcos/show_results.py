import torch
from utils.model_utils import get_model
from utils.checkpoint_utils import load_prog
from utils.data_utils import getIBEMClasses
from utils.display_utils import plot_results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}.'.format(device))
    IBEM_CLASSES = getIBEMClasses()

    model = get_model(device, num_classes=len(IBEM_CLASSES))
    optim=torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = "checkpoints/ibem_mod/checkpoint_epoch50_run.pth"

    if checkpoint_path is not None:
        _, _, _, train_loss, val_prec, val_rec, val_mAP = load_prog(checkpoint_path, model, optim)

    plot_results(train_loss, "Loss", "Training Loss per Epoch")
    plot_results(val_prec, "Precision", "Validation Precision per Epoch")
    plot_results(val_prec, "Recall", "Validation Recall per Epoch")
    plot_results(val_prec, "mAP", "Validation mAP per Epoch")