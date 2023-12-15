import torchvision
if __name__ == "__main__":
    torchvision.models.detection.fcos_resnet50_fpn(progress=False, num_classes=4)