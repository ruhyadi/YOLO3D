from ast import arg
import gdown
import argparse

parser = argparse.ArgumentParser(description='Download weights model')
parser.add_argument('--weights', type=str, help='Weights name')

args = parser.parse_args()

def get_weights(weights):
    weights_list = {
        'resnet': '1Bw4gUsRBxy8XZDGchPJ_URQjbHItikjw',
        'resnet18': '1k_v1RrDO6da_NDhBtMZL5c0QSogCmiRn',
        'vgg11': '1vZcB-NaPUCovVA-pH-g-3NNJuUA948ni'
    }

    url = f"https://drive.google.com/uc?id={weights_list[weights]}"
    output = f"./{weights}.pkl"
    gdown.download(url, output, quiet=False)

get_weights(args.weights)