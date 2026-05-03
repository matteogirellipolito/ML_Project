import os
import glob
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from erfnet import ERFNet
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import scipy.special

NUM_CLASSES = 20

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    output_dir = "outputs_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    continue
            else:
                own_state[name].copy_(param)
        return model

    weightspath = args.loadDir + args.loadWeights
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model LOADED")
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print("Processing:", path)

        image = Image.open(path).convert('RGB')
        image_np = np.array(image)

        tensor_img = input_transform(image).unsqueeze(0).float().cuda()

        with torch.no_grad():
            result = model(tensor_img)

        logits = result.squeeze(0).data.cpu().numpy()

        maxlogit = -np.max(logits, axis=0)

        probs = scipy.special.softmax(logits, axis=0)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-10)

        maxlogit_norm = normalize(maxlogit)
        entropy_norm = normalize(entropy)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(maxlogit_norm, cmap='jet')
        axs[1].set_title("MaxLogit")
        axs[1].axis('off')

        axs[2].imshow(entropy_norm, cmap='jet')
        axs[2].set_title("MaxEntropy")
        axs[2].axis('off')

        filename = os.path.basename(path).split('.')[0]
        save_path = os.path.join(output_dir, f"{filename}_heatmap.png")

        plt.savefig(save_path)
        plt.close()

    print(f"\nSaved all heatmaps in: {output_dir}")

if __name__ == '__main__':
    main()