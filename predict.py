# Predict flower name from an image with predict.py along with the probability of that name. 
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#
#    Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#        Return top K most likely classes: python predict.py input checkpoint --top_k 3
#        Use a mapping of categories to real names: 
#            python predict.py input checkpoint --category_names cat_to_name.json
#        Use GPU for inference: python predict.py input checkpoint --gpu

import argparse
import numpy as np
from PIL import Image
import torch
import json
from torchvision import models
from util import datestr, build_sequential

# Parse input arguments
parser = argparse.ArgumentParser(
    description='Predict flower name and predicted probability from an image.'
)

parser.add_argument('image_path', type=str, help='path to input image file')
parser.add_argument('checkpoint_path', type=str, help='path to checkpoint file')
parser.add_argument('--top_k', type=int, required=False, nargs='?', default=5, metavar="K", help='number of top results to return (default=%(default)s)')
parser.add_argument('--category_names', type=str, required=False, nargs='?', default='cat_to_name.json', metavar='JSON_FILEPATH', help='path to category-to-names JSON file (default=\'%(default)s\')')
parser.add_argument('--gpu', action='store_const', const='use_gpu', default=False, help='use gpu')

args = parser.parse_args()
print("args = {}".format(args))

# Use GPU if it's available and specified
if args.gpu and torch.cuda.is_available():
    devicename = "cuda"
elif args.gpu:
    print("Warning: --gpu specified but cuda is not available.")
    devicename = "cpu"
else:
    devicename = "cpu"
    
device = torch.device(devicename)
print("Predicting with {}...".format(device))

# Label mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Load the model and checkpoint
def load_checkpoint(filepath):
    """Loads the checkpoint and rebuilds the model"""
    # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349 <-- AttributeError: 'str' object has no attribute '__module__'
    checkpoint = torch.load(filepath, map_location='cpu')#, map_location=lambda storage, location: 'cpu')
    
    if checkpoint['args'].arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        n_input = model.classifier[0].in_features
    elif checkpoint['args'].arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        n_input = model.classifier[0].in_features
    elif checkpoint['args'].arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        n_input = model.classifier.in_features
    else:
        raise ValueError('Encountered unexpected \'arch\' argument')
    
    # https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists
    if 'checkpoint[\'args\'].n_input' not in locals():
        checkpoint['args'].n_input = n_input
        print("n_input = {}".format(checkpoint['args'].n_input))
    
    model.classifier = build_sequential(checkpoint)    
    model.classifier.load_state_dict(checkpoint['model_classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model
    
model = load_checkpoint(args.checkpoint_path)
print("Loaded {} model.".format(type(model)))
print(model)

for param in model.parameters():
    param.requires_grad = False

# Read and pre-process input image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a <s>Numpy array</s> PyTorch tensor!
    '''

    # DONE: Process a PIL image for use in a PyTorch model
    min_width = 256
    min_height = 256
    crop_width = 224
    crop_height = 224
    
    pil_image = Image.open(image)
    #print(pil_image)
    #display(pil_image)

    width, height = pil_image.size
    resize_ratio = max(min_width / width, min_height / height)
    new_width = round(width*resize_ratio)
    new_height = round(height*resize_ratio)
    pil_image = pil_image.resize((new_width, new_height))
    #print(pil_image)
    #display(pil_image)
    
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    box_left = (new_width - crop_width) / 2.0
    box_right = (new_width + crop_width) / 2.0
    box_top = (new_height - crop_width) / 2.0
    box_bottom = (new_height + crop_width) / 2.0

    pil_image = pil_image.crop((box_left, box_top, box_right, box_bottom))
    #print((box_left, box_top, box_right, box_bottom))
    #print(pil_image)
    #display(pil_image)
    
    np_image = np.array(pil_image)/255.0
    norm_means = np.array([0.485, 0.456, 0.406])
    norm_stdevs = np.array([0.229, 0.224, 0.225])
    normalized_np_image = (np_image - norm_means) / norm_stdevs
    
    # Put the color channel first for PyTorch
    normalized_np_image = normalized_np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(normalized_np_image)

# Perform class prediction including a model forward pass
def predict(image_path, model, devicename, topk=5):
    ''' Predict    the class (or classes) of an image using a trained deep learning model.
    urns the top 𝐾 most likely classes along with the probabilities. E.g. usage:
    
        probs, classes = predict(image_path, model)
    
    There's some ambiguity in the instructions, but I presumed that the function is 
    actually meant to return the class /indices/ for the "classes" return variable.
    '''
        
    # DONE: Implement the code to predict the class from an image file
    pytorch_image = process_image(image_path)
    torch.save(pytorch_image, 'pytorch_image.pth')

    model.eval()
    model = model.to(devicename)
    pytorch_image = pytorch_image.unsqueeze_(0).float().to(device)

    #print("predicting...")
    #print(pytorch_image)
    #print(pytorch_image.shape)
    # Get the log probabilities from a forward pass
    with torch.no_grad():
        # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
        # https://stackoverflow.com/questions/49407303/runtimeerror-expected-object-of-type-torch-doubletensor-but-found-type-torch-fl
        log_ps = model.forward(pytorch_image)
    
    # Get the class probabilities
    ps = torch.exp(log_ps)

    # Get the top results
    top_p, top_idx = ps.topk(topk, dim=1)

    #print(top_p)
    #print(top_idx)
    
    # https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
    # TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    #print(top_class)

    # https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
    # TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    top_idx_list = top_idx.cpu().numpy().squeeze().tolist()
    
    #print(top_idx_list)
    #print(type(top_idx_list))
    #if top_idx_np.ndim == 0:
    #    #print('trying this')
    #    top_idx_list = [top_idx_np]
    #else:
    #    #print('trying that')
    #    top_idx_list = top_idx_np.tolist()
    
    #print(model.class_to_idx)
    #print(top_idx_list)
    # http://stupidpythonideas.blogspot.com/2014/07/reverse-dictionary-lookup-and-more-on.html
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    top_class = [idx_to_class[x] for x in top_idx_list]
    
    return top_p, top_idx_list, top_class

# Call the classifier to produce a prediction
imgpath = args.image_path
topk = args.top_k
imgpath_parts = imgpath.split('/')
imgpath_class = imgpath_parts[-2]
#print(imgpath)
#print(imgpath_parts)
#print(imgpath_class)

pytorch_image = process_image(imgpath)

# Predict the type of the input image
# https://discuss.pytorch.org/t/type-mismatch-on-model-when-using-gpu/11409
probs, indices, classes = predict(imgpath, model, devicename, topk)

probs_np = probs.cpu().numpy().squeeze()

print(probs)
print(indices)
print(classes)

# Print the top K results
print("\ninput was \'{}\' (a {})".format(imgpath, cat_to_name[imgpath_class]))

for i in range(topk):
    print("    prediction #{} of top {} (probability {:4.1f}%): class \'{}\' (index {}) ({})".format(
        i + 1, topk, 100.0*probs_np[i], classes[i], indices[i], cat_to_name[str(classes[i])]))
