import gdown

# TODO: add gdrive link
url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
output = "resnet_10.pkl"
gdown.download(url, output, quiet=False)