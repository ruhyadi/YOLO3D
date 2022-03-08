import gdown

url = "https://drive.google.com/uc?id=1Bw4gUsRBxy8XZDGchPJ_URQjbHItikjw"
output = "./resnet_10.pkl"
gdown.download(url, output, quiet=False)